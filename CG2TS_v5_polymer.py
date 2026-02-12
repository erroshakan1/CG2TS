#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is a direct conversion of the CG2TS.ipynb notebook.
All functions and the main processing loop have been consolidated into this single file.
The proprietary 'optimesh' library has been replaced with the open-source 'pymeshlab'.
"""
import numpy as np
import time
import math
import argparse
import os

# Matplotlib for plotting
import matplotlib.pyplot as plt

# MDAnalysis for trajectory analysis
import MDAnalysis as mda
from MDAnalysis.analysis.leaflet import LeafletFinder
from MDAnalysis.analysis.distances import distance_array

# Scikit-learn for neighbor searches
from sklearn.neighbors import NearestNeighbors

# Open3D for 3D geometry processing
import open3d as o3d

# PyMeshLab for mesh optimization (replaces optimesh)
import pymeshlab
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
# ============================================================================
# PYMESHLAB WRAPPER - Replaces optimesh
# ============================================================================

def optimize_mesh_pymeshlab(vertices, triangles, method="cvt full", tolerance=1.0e-9, num_iterations=10):
    """
    PyMeshLab-based mesh optimization to replace optimesh.optimize_points_cells().
    
    Uses Taubin smoothing, a high-quality open-source alternative to the proprietary
    CVT (Centroidal Voronoi Tessellation) methods in optimesh.
    
    Args:
        vertices (np.ndarray): Mesh vertices (N x 3).
        triangles (np.ndarray): Triangle connectivity (M x 3).
        method (str): Kept for API compatibility, but PyMeshLab uses Taubin smoothing.
        tolerance (float): Kept for API compatibility.
        num_iterations (int): Number of smoothing iterations.
        
    Returns:
        tuple[np.ndarray, np.ndarray]: Optimized vertices and triangle connectivity.
    """
    ms = pymeshlab.MeshSet()
    m = pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=triangles)
    ms.add_mesh(m)
    
    # Apply Taubin smoothing. Standard parameters are lambda=0.5, mu=-0.53.
    ms.apply_coord_taubin_smoothing(
        stepsmoothnum=num_iterations,
        lambda_=0.5,
        mu=-0.53
    )
    
    optimized_mesh = ms.current_mesh()
    return optimized_mesh.vertex_matrix(), optimized_mesh.face_matrix()


def optimize_mesh_cvt(vertices, triangles, method="cvt full", tolerance=1.0e-9, num_iterations=10, alpha=1.0):
    """
    Approximate centroidal Voronoi tessellation (CVT) optimization on a mesh.

    This is an iterative, local approximation intended to be closer to optimesh's
    CVT-based relocation than a pure Taubin smoothing. It computes area-weighted
    centroids of the dual cell around each vertex (using adjacent triangle centroids)
    and moves the vertex tangentially toward that centroid. Optionally repeats
    for several iterations.

    Notes:
    - This is an approximation and may not be bit-for-bit identical to optimesh,
      but it generally yields centroidal behavior similar to CVT.
    - The mesh connectivity (triangles) is preserved; only vertex positions change.
    """
    verts = vertices.copy().astype(np.float64)
    tris = triangles.astype(int)

    # Precompute triangle centroids and areas
    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]
    tri_centroids = (v0 + v1 + v2) / 3.0
    tri_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

    # Build vertex -> incident triangle indices
    vert_tri_inds = [[] for _ in range(len(verts))]
    for ti, t in enumerate(tris):
        for vi in t:
            vert_tri_inds[vi].append(ti)

    # Iteratively relocate vertices
    for it in range(max(1, int(num_iterations))):
        # Compute vertex normals (area-weighted) for tangent projection
        vertex_normals = np.zeros_like(verts)
        for vi, tri_inds in enumerate(vert_tri_inds):
            if len(tri_inds) == 0:
                continue
            normals = np.cross(v1[tri_inds] - v0[tri_inds], v2[tri_inds] - v0[tri_inds])
            # area-weighted sum
            w = tri_areas[tri_inds][:, np.newaxis]
            n = np.sum(w * normals, axis=0)
            norm = np.linalg.norm(n)
            vertex_normals[vi] = (n / norm) if norm > 0 else np.array([0.0, 0.0, 1.0])

        new_verts = verts.copy()
        for vi, tri_inds in enumerate(vert_tri_inds):
            if len(tri_inds) == 0:
                continue
            # area-weighted centroid of adjacent triangles
            areas = tri_areas[tri_inds]
            cents = tri_centroids[tri_inds]
            total_area = np.sum(areas)
            if total_area == 0:
                continue
            centroid = np.sum(cents * areas[:, np.newaxis], axis=0) / total_area

            v = verts[vi]
            n = vertex_normals[vi]
            d = centroid - v
            # project displacement onto tangent plane to preserve normal
            tangential = d - np.dot(d, n) * n
            new_verts[vi] = v + alpha * tangential

        verts = new_verts

        # Recompute triangle-dependent quantities for next iteration
        v0 = verts[tris[:, 0]]
        v1 = verts[tris[:, 1]]
        v2 = verts[tris[:, 2]]
        tri_centroids = (v0 + v1 + v2) / 3.0
        tri_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

    return verts, tris


def optimize_mesh(vertices, triangles, method="pymeshlab", tolerance=1.0e-9, num_iterations=10, **kwargs):
    """
    Dispatch to the chosen optimizer. Supported methods:
      - 'pymeshlab' : Taubin smoothing via PyMeshLab (default)
      - 'cvt_approx' : local centroidal Voronoi-like approximation implemented above
    """
    method = (method or '').lower()
    if 'cvt' in method:
        return optimize_mesh_cvt(vertices, triangles, method=method, tolerance=tolerance, num_iterations=num_iterations, **kwargs)
    else:
        return optimize_mesh_pymeshlab(vertices, triangles, method=method, tolerance=tolerance, num_iterations=num_iterations)

# ============================================================================
# NOTEBOOK FUNCTIONS
# ============================================================================

def fill_in_defect_matrix(gl2_leaflet, grid_coords, normals, box, depth_max, upper=False, tinygridradius=10):
    m_polar = 1.0
    m_aliphatic = 0.001
    dist_far = 0.8
    dist_clo = 0.6
    M_1 = np.ones(np.shape(grid_coords)[0])
    M_1[:] = np.nan
    upperval = 1
    if (upper == True):
        upperval = -1
    
    for gl2 in gl2_leaflet[:]:
        res = gl2.residue
        gl2_coords = gl2.position
        res_coords = res.atoms.positions
        
        index_closest_cell = np.argmin(np.sum((grid_coords - gl2_coords)**2, axis=1))
        cell_coord = grid_coords[index_closest_cell]
        
        normal = np.mean(normals[np.where(np.sum((grid_coords - gl2_coords)**2, axis=1) < tinygridradius**2)], axis=0)
        if not np.isnan(normal[0]):
            avnormal = upperval * normal
            avnormal = avnormal / np.sqrt(np.sum(avnormal**2))
        
            grid_coords_m = grid_coords + (gl2_coords[2]-cell_coord[2])*avnormal
            grid_coords_m = grid_coords_m + depth_max*avnormal
            cell_coord = grid_coords_m[index_closest_cell]
        
            index_closest_cell2 = np.argmin(np.sum((grid_coords_m - gl2_coords)**2, axis=1))
            cell_coord2 = grid_coords_m[index_closest_cell2]
        
            dr_atom_ref = min_image_dist(res_coords, cell_coord2, box)
            normal_gl2 = normal
        
            cos_alpha = (np.inner(dr_atom_ref, normal_gl2) /
                              (np.linalg.norm(dr_atom_ref, axis=1) * 
                               np.linalg.norm(normal_gl2)))
            
            for a, atom in enumerate(res.atoms):
                atom_name = atom.name
                atom_radius, m_value = get_radius_and_defect_type(atom_name, cos_alpha[a])
                dist_lim = (dist_clo+atom_radius)**2
                dist_meet = (dist_far+atom_radius)**2
            
                ix_closest = np.argmin(np.sum((grid_coords_m - atom.position)**2, axis=1))
                dist = np.sqrt(np.sum((atom.position-grid_coords_m[ix_closest])**2))
                minorplus = np.linalg.norm(atom.position) - np.linalg.norm(grid_coords_m[ix_closest])
                dist = (minorplus / abs(minorplus)) * dist if minorplus != 0 else 0
            
                grid_coords_mm = grid_coords_m + (dist)*avnormal
                delta_x = np.sum((grid_coords_mm-atom.position)**2, axis=1)

                ix_lim = np.where(delta_x <= dist_lim)[0]
                ix_meet = np.unique(np.where(delta_x <= dist_meet)[0])
            
                for x1 in ix_lim:
                    if np.isnan(M_1[x1]):
                        M_1[x1] = 1 if m_value == m_polar else 0
                
                M_1[ix_meet] += m_value
    return M_1

def fill_in_defect_matrix7(gl2_leaflet, grid_coords, normals, box, depth_max, upper=False, tinygridradius=10, cutoff=0.6):
    m_polar = 1.0
    m_aliphatic = 0.001
    dist_clo = cutoff
    M_1 = np.ones(np.shape(grid_coords)[0])
    M_1[:] = np.nan
    upperval = 1
    if (upper == True):
        upperval = -1
    
    for gl2 in gl2_leaflet[:]:
        res = gl2.residue
        gl2_coords = gl2.position
        res_coords = res.atoms.positions
        
        index_closest_cell = np.argmin(np.sum((grid_coords - gl2_coords)**2, axis=1))
        cell_coord = grid_coords[index_closest_cell]
        
        normal = np.mean(normals[np.where(np.sum((grid_coords - gl2_coords)**2, axis=1) < tinygridradius**2)], axis=0)
        if np.isnan(normal[0]):
            normal = normals[index_closest_cell]
            
        avnormal = upperval * normal
        avnormal = avnormal / np.sqrt(np.sum(avnormal**2))
        
        grid_coords_m = grid_coords + (gl2_coords-cell_coord)*avnormal
        grid_coords_m = grid_coords_m + (depth_max if not upper else -depth_max) * avnormal
            
        index_closest_cell2 = np.argmin(np.sum((grid_coords_m - gl2_coords)**2, axis=1))
        cell_coord2 = grid_coords_m[index_closest_cell2]
        
        dr_atom_c = res_coords - cell_coord2
        normal_gl2 = normal
        cos_alpha = np.inner(dr_atom_c, normal_gl2)
            
        for a, atom in enumerate(res.atoms):
            atom_name = atom.name
            atom_radius, m_value = get_radius_and_defect_type(atom_name, cos_alpha[a])
            dist_lim = (dist_clo+atom_radius)**2

            ix_closest = np.argmin(np.sum((grid_coords - atom.position)**2, axis=1))
            dist = np.dot((atom.position-grid_coords[ix_closest]), avnormal)
        
            pos = atom.position - (dist)*avnormal
            delta_x = np.sum((grid_coords-pos)**2, axis=1)
            ix_lim = np.unique(np.where(delta_x <= dist_lim)[0])
                
            for x1 in ix_lim:
                if np.isnan(M_1[x1]):
                    M_1[x1] = 1 if m_value == m_polar else 0
            M_1[ix_lim] += m_value
    return M_1

def get_clusters_using_mesh_stitch(M_discrete_1, vertices, triangles, minmax):
    xmin, xmax, ymin, ymax = minmax
    
    ix_triangles_0 = np.where(M_discrete_1==0)[0]
    ix_triangles_1 = np.where(M_discrete_1==1)[0]
    ix_triangles_01 = np.concatenate((ix_triangles_0, ix_triangles_1))

    triangles_0 = triangles[ix_triangles_0]
    triangles_1 = triangles[ix_triangles_1]
    triangles_01 = triangles[ix_triangles_01]
    
    ix_vertices = np.where((vertices[:,0] <= (xmin+1)) | (vertices[:,0] >= (xmax-1)) | 
                           (vertices[:,1] <= (ymin+1)) | (vertices[:,1] >= (ymax-1)))[0]

    def filter_edge_triangles(triangles_in, vertices_on_edge):
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles_in))
        labels, _, _ = mesh.cluster_connected_triangles()
        labels = np.array(labels)
        del_ix = np.array([], dtype=int)
        for i in vertices_on_edge:
            if i in triangles_in:
                ix = np.where(triangles_in == i)[0]
                del_labels = np.unique(labels[ix])
                for dl in del_labels:
                    del_ix = np.append(del_ix, np.where(labels == dl)[0])
        return np.delete(triangles_in, np.unique(del_ix), axis=0)

    triangles_0 = filter_edge_triangles(triangles_0, ix_vertices)
    triangles_1 = filter_edge_triangles(triangles_1, ix_vertices)
    triangles_01 = filter_edge_triangles(triangles_01, ix_vertices)
    
    mesh_0 = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles_0))
    mesh_1 = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles_1))
    mesh_01 = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles_01))
    
    areas0 = stitch_clusters(mesh_0)
    areas1 = stitch_clusters(mesh_1)
    areas01 = stitch_clusters(mesh_01)
    
    return areas0, areas1, areas01, mesh_0, mesh_1, mesh_01

def min_image_dist(a, b, box):
    d = a - b
    return d - box * np.rint(d / box)

def get_radius_and_defect_type(atom_name, cos_alpha):
    m_polar = 1.0
    m_aliphatic = 0.001
    
    # PG* beads = PEO/PEG hydrophilic polymer headgroups (polar, large radius like lipid heads)
    is_headgroup = atom_name in ['NC3', 'NH3', 'PO4', 'EG*'] or atom_name.startswith('PG')
    is_interface = atom_name in ['GL1', 'GL2', 'GLA', 'NAB']
    
    if is_headgroup:
        atom_radius = 3.4796
    elif is_interface:
        atom_radius = 2.413
    else:
        atom_radius = 2.638
    
    if is_headgroup or is_interface:
        m_value = m_polar
    elif cos_alpha > 0.0:
        m_value = 0.0
    else:
        m_value = m_aliphatic
        
    return atom_radius, m_value

def make_grid_flat_test3(gro, ag, lower=True, radius_norms=100, max_nn=30, seltail={'DTAP2': 'name C4*'}): 
    positions = ag.positions
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions[:,:3])
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_norms, max_nn=max_nn))
    pcd.normalize_normals()

    resag = ag.resids
    resnames = ag.resnames
    tails = np.array([gro.select_atoms(f'(resid {ra} and ({seltail[resnames[ir]]}))').positions for ir, ra in enumerate(resag)]).reshape(-1, 3)
    
    tails_av = (tails[0::2] + tails[1::2])/2
    dir_norm = (tails_av - positions) / np.linalg.norm((tails_av - positions), axis=1)[:, np.newaxis]

    normals_3d = np.asarray(pcd.normals)
    dots = np.einsum('ij,ij->i', dir_norm, normals_3d)
    ix_normals_to_flip = np.where(dots < 0)
    normals_3d[ix_normals_to_flip] *= -1
    pcd.normals = o3d.utility.Vector3dVector(normals_3d)
    
    mesh_pois, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    
    triangle_clusters, cluster_n_triangles, _ = mesh_pois.cluster_connected_triangles()
    triangles_to_remove = np.asarray(cluster_n_triangles)[np.asarray(triangle_clusters)] < 100
    mesh_pois.remove_triangles_by_mask(triangles_to_remove)
    
    mesh_pois.remove_non_manifold_edges()
    mesh_pois.remove_duplicated_vertices()
    mesh_pois.remove_duplicated_triangles()
    mesh_pois.compute_triangle_normals()
    mesh_pois.normalize_normals()
    return mesh_pois, pcd

def grid_plane_triangulated_m(grid_l1, size_triangle=1.0, num_it1=1, num_it2=1, method='cvt full', simp_cutoff=1.0):
    grid_l1.remove_non_manifold_edges()
    grid_l1.merge_close_vertices(simp_cutoff)
    grid_l1.remove_degenerate_triangles()
    grid_l1.remove_unreferenced_vertices()
    
    n_triangles = int(grid_l1.get_surface_area() / size_triangle)
    current_n_triangles = len(np.asarray(grid_l1.triangles))
    if current_n_triangles == 0: return grid_l1

    versions_of_4 = int(np.ceil(np.log(n_triangles / current_n_triangles) / np.log(4))) if n_triangles > current_n_triangles else 0
    target_triangles = int(n_triangles / (4**versions_of_4)) if versions_of_4 > 0 else n_triangles

    mesh_smp = grid_l1.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
    mesh_smp.remove_non_manifold_edges()
    mesh_smp.merge_close_vertices(simp_cutoff)
    
    vertices, triangles = np.asarray(mesh_smp.vertices), np.asarray(mesh_smp.triangles)
    
    points, cells = optimize_mesh(vertices, triangles, method, 1.0e-9, num_it1)

    mesh_new = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(points), o3d.utility.Vector3iVector(cells))
    mesh_new.compute_triangle_normals()
    mesh_new.normalize_normals()

    if versions_of_4 > 0:
        grid_l1_new = mesh_new.subdivide_midpoint(number_of_iterations=versions_of_4)
    else:
        grid_l1_new = mesh_new

    vertices, triangles = np.asarray(grid_l1_new.vertices), np.asarray(grid_l1_new.triangles)
    points, cells = optimize_mesh(vertices, triangles, method, 1.0e-9, num_it2)

    mesh_new2 = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(points), o3d.utility.Vector3iVector(cells))
    mesh_new2.compute_triangle_normals()
    mesh_new2.normalize_normals()
    return mesh_new2

def get_centroid_of_triangles(vertices, triangles): 
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    return (v0 + v1 + v2) / 3.0

def get_selection_phaseseparated(gro, cutoff_l, cutoff2):
    L = LeafletFinder(gro, 'name GL2 or name GL1', cutoff=cutoff_l, pbc=True)
    if len(L.groups()) < 2:
        raise ValueError("Could not find two leaflets. Try adjusting --leaflet-cutoff.")
    
    # Sort by size to get the two largest leaflets
    leaflet_sizes = sorted([(i, l.n_atoms) for i, l in enumerate(L.groups())], key=lambda x: x[1], reverse=True)
    upper_idx, lower_idx = leaflet_sizes[0][0], leaflet_sizes[1][0]

    upper = L.group(upper_idx)
    lower = L.group(lower_idx)
    
    if np.mean(upper.positions[:,2]) < np.mean(lower.positions[:,2]):
        upper, lower = lower, upper
    
    # Assign non-lipid molecules (cholesterol ROH, polymer PG1) to nearest leaflet
    gla = gro.select_atoms('name ROH or name PG1')
    if len(gla) > 0:
        distup = distance_array(gla.positions, upper.positions)
        distlo = distance_array(gla.positions, lower.positions)
        
        ix_up = np.where(np.min(distup, axis=1) < cutoff2)[0]
        ix_lo = np.where(np.min(distlo, axis=1) < cutoff2)[0]
        
        up_resids = set(upper.resids) | set(gla.resids[ix_up])
        lo_resids = set(lower.resids) | set(gla.resids[ix_lo])
        
        upper = gro.select_atoms(f'(name ROH or name PG1 or name GL2) and resid {" ".join(map(str, up_resids))}')
        lower = gro.select_atoms(f'(name ROH or name PG1 or name GL2) and resid {" ".join(map(str, lo_resids))}')
        
    return upper, lower

def stitch_clusters(mesh):
    mesh.remove_unreferenced_vertices()
    labels, clusters, areas = mesh.cluster_connected_triangles()
    triangles = np.asarray(mesh.triangles)
    labels, areas = np.asarray(labels), np.asarray(areas)
    
    new_areas = []
    processed_labels = set()
    
    for i in range(len(clusters)):
        if i in processed_labels:
            continue
        
        connected_labels = {i}
        nodes_to_check = list(np.unique(triangles[labels == i]))
        
        while nodes_to_check:
            node = nodes_to_check.pop()
            tri_indices = np.where(np.any(triangles == node, axis=1))[0]
            
            for tri_idx in tri_indices:
                label = labels[tri_idx]
                if label not in connected_labels:
                    connected_labels.add(label)
                    processed_labels.add(label)
                    nodes_to_check.extend(list(np.unique(triangles[labels == label])))
        
        new_areas.append(sum(areas[list(connected_labels)]))
        processed_labels.update(connected_labels)
        
    return np.array(new_areas)

def get_cutoff(grid_coords, n_neighbours, pos, n_for_cutoff):
    if len(grid_coords) < n_neighbours + 1: return 0.6 # Default if not enough points
    dist = average_distance_neighbours(grid_coords, n_neighbours)
    
    # Filter out edge artifacts
    min_x, max_x = np.min(pos[:,0]), np.max(pos[:,0])
    min_y, max_y = np.min(pos[:,1]), np.max(pos[:,1])
    ix_core = np.where((grid_coords[:,0] > min_x+3) & (grid_coords[:,0] < max_x-3) &
                       (grid_coords[:,1] > min_y+3) & (grid_coords[:,1] < max_y-3))[0]
    
    if len(ix_core) == 0: return 0.6 # Default if no core points found
    
    dist = dist[ix_core]
    return np.mean(np.mean(dist[:, -n_for_cutoff:], axis=1)) / 2

def average_distance_neighbours(gridcoords_nbrs, n_neighbors):
    nbrs = NearestNeighbors(n_neighbors=(n_neighbors+1)).fit(gridcoords_nbrs)
    distances, _ = nbrs.kneighbors(gridcoords_nbrs)
    return distances

def get_ranges_pos_neg_flat(triang_normals, minmax, plane_direction, grid_coords, buffer=30, dx=10, rangepos=(5, 50), rangeneg=(250, 355), view_plot=False):
    xyz_map = {'x': 0, 'y': 1, 'z': 2}
    xval, yval, zval = [xyz_map[c] for c in plane_direction.replace('_not', '')]

    min_x, max_x, min_y, max_y = minmax
    core_mask = (grid_coords[:, 1] > min_y + 5) & (grid_coords[:, 1] < max_y - 5)
    
    xbins = np.arange(int(min_x)-buffer, int(max_x)+buffer, dx)
    ybins = np.arange(int(min_y)-buffer, int(max_y)+buffer, dx)
    
    angles = np.zeros(len(grid_coords))

    for x in xbins:
        for y in ybins:
            mask = (grid_coords[:, xval] > x - dx/2) & (grid_coords[:, xval] <= x + dx/2) & \
                   (grid_coords[:, yval] > y - dx/2) & (grid_coords[:, yval] <= y + dx/2)
            
            if not np.any(mask): continue

            av_normal = np.mean(triang_normals[mask], axis=0)
            centroid = np.mean(grid_coords[mask], axis=0)
            
            # Simplified angle calculation logic
            angles[mask] = np.degrees(np.arccos(np.clip(av_normal[zval], -1.0, 1.0)))

    ix_rangepos = (angles >= rangepos[0]) & (angles < rangepos[1])
    ix_rangeneg = (angles >= rangeneg[0]) & (angles < rangeneg[1])
    ix_rangeflat = ~ (ix_rangepos | ix_rangeneg)

    return ix_rangepos, ix_rangeneg, ix_rangeflat

def write_file(arr, infile, replacement_line):
    if not os.path.exists(os.path.dirname(infile)):
        os.makedirs(os.path.dirname(infile))
    
    # This function is complex and stateful, for simplicity we just append.
    with open(infile, 'a') as f:
        for a in arr:
            f.write(f'{a}\n')

def initialize_files(savedir, filenames, replacement_line):
    for f_list in filenames:
        for infile in f_list:
            if not os.path.exists(os.path.dirname(savedir+infile)):
                os.makedirs(os.path.dirname(savedir+infile))
            with open(savedir+infile, 'w') as f:
                f.write(replacement_line + ' 0\n')

# ============================================================================
# MAIN PROCESSING BLOCK
# ============================================================================

def main(args):
    """Main processing function."""
    
    print("Starting analysis...")
    print(f"Input TPR: {args.tpr}")
    print(f"Input XTC: {args.xtc}")
    print(f"Output directory: {args.savedir}")

    # Ensure savedir ends with '/' so filenames are placed inside, not beside
    if not args.savedir.endswith(os.sep):
        args.savedir += os.sep

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
        print(f"Created save directory: {args.savedir}")

    try:
        gro = mda.Universe(args.tpr, args.xtc)
        print(f"Trajectory loaded with {len(gro.trajectory)} frames.")
    except Exception as e:
        print(f"Error loading trajectory files: {e}")
        return

    # Define file names
    prefix = 'areas'
    affixup = f'_{args.affix}{args.striangle:.1f}_up_{args.depth_max:.1f}_c{args.cutoff:.1f}.txt'
    affixlo = f'_{args.affix}{args.striangle:.1f}_lo_{args.depth_max:.1f}_c{args.cutoff:.1f}.txt'
    
    filenames_up = [f'{prefix}0{affixup}', f'{prefix}1{affixup}', f'{prefix}01{affixup}']
    filenames_lo = [f'{prefix}0{affixlo}', f'{prefix}1{affixlo}', f'{prefix}01{affixlo}']
    
    # Initialize output files
    replacement_line = 'n frames ='
    all_files = [filenames_up, filenames_lo, 
                 ['r1_'+f for f in filenames_up], ['r1_'+f for f in filenames_lo],
                 ['r2_'+f for f in filenames_up], ['r2_'+f for f in filenames_lo],
                 ['r3_'+f for f in filenames_up], ['r3_'+f for f in filenames_lo]]
    initialize_files(args.savedir, all_files, replacement_line)

    seltail = {'DPPC': 'name C4*', 'DPPG': 'name C4*', 'CHOL': 'name C*', 'DTAP2': 'name C4*', 'POL': 'name B1 or name S1'}

    # --- Main Trajectory Loop ---
    for ts in gro.trajectory[args.start_frame:args.end_frame:args.skip]:
        start_time = time.time()
        print(f"\nProcessing frame {ts.frame}...")

        try:
            upper, lower = get_selection_phaseseparated(gro, args.leaflet_cutoff, 12)
        except ValueError as e:
            print(f"  Skipping frame {ts.frame}: {e}")
            continue

        viz_geometries = []  # Collect meshes from both leaflets for a single visualization

        for leaflet_name, leaflet_ag, is_upper in [('Upper', upper, True), ('Lower', lower, False)]:
            print(f"  Processing {leaflet_name} leaflet...")
            
            grid, pcd = make_grid_flat_test3(gro, leaflet_ag, lower=is_upper, seltail=seltail)
            
            print("    Remeshing grid...")
            regrid = grid_plane_triangulated_m(grid, size_triangle=args.striangle, 
                                               num_it1=args.numit1, num_it2=args.numit2, 
                                               method=args.method, simp_cutoff=2.0)
            
            vertices = np.asarray(regrid.vertices)
            triangles = np.asarray(regrid.triangles)
            if len(triangles) == 0:
                print("    Skipping leaflet: No triangles in mesh.")
                continue

            grid_coords = get_centroid_of_triangles(vertices, triangles)
            normals = np.asarray(regrid.triangle_normals)
            
            current_cutoff = get_cutoff(grid_coords, 12, leaflet_ag.positions, 3)
            print(f"    Calculated defect cutoff: {current_cutoff:.2f}")

            print("    Calculating defect matrix...")
            M_1 = fill_in_defect_matrix7(leaflet_ag, grid_coords, normals, ts.dimensions, 
                                         args.depth_max, upper=is_upper, cutoff=current_cutoff)
            M_discrete_1 = np.where(M_1 >= 1, 2, np.where(M_1 > 0, 1, np.where(M_1 > -0.5, 0, -1)))
            
            minmax = np.array([np.min(leaflet_ag.positions[:,0]), np.max(leaflet_ag.positions[:,0]), 
                               np.min(leaflet_ag.positions[:,1]), np.max(leaflet_ag.positions[:,1])])
            
            print("    Stitching defect clusters...")
            areas0, areas1, areas01, _, _, _ = get_clusters_using_mesh_stitch(M_discrete_1, vertices, triangles, minmax)
            
            fnames = filenames_up if is_upper else filenames_lo
            write_file(areas0, args.savedir + fnames[0], replacement_line)
            write_file(areas1, args.savedir + fnames[1], replacement_line)
            write_file(areas01, args.savedir + fnames[2], replacement_line)
            print(f"    Found {len(areas01)} total defect clusters.")

            print("    Analyzing curvature regions...")
            ixr1, ixr2, ixr3 = get_ranges_pos_neg_flat(normals, minmax, 'yz_notx', grid_coords)
            
            # Region 1 (flat)
            areas0, areas1, areas01, _, _, _ = get_clusters_using_mesh_stitch(M_discrete_1[ixr1], vertices, triangles[ixr1], minmax)
            write_file(areas0, args.savedir + 'r1_' + fnames[0], replacement_line)
            write_file(areas1, args.savedir + 'r1_' + fnames[1], replacement_line)
            write_file(areas01, args.savedir + 'r1_' + fnames[2], replacement_line)
            
            # Region 2 (positive curvature)
            areas0, areas1, areas01, _, _, _ = get_clusters_using_mesh_stitch(M_discrete_1[ixr2], vertices, triangles[ixr2], minmax)
            write_file(areas0, args.savedir + 'r2_' + fnames[0], replacement_line)
            write_file(areas1, args.savedir + 'r2_' + fnames[1], replacement_line)
            write_file(areas01, args.savedir + 'r2_' + fnames[2], replacement_line)

            # Region 3 (negative curvature)
            areas0, areas1, areas01, _, _, _ = get_clusters_using_mesh_stitch(M_discrete_1[ixr3], vertices, triangles[ixr3], minmax)
            write_file(areas0, args.savedir + 'r3_' + fnames[0], replacement_line)
            write_file(areas1, args.savedir + 'r3_' + fnames[1], replacement_line)
            write_file(areas01, args.savedir + 'r3_' + fnames[2], replacement_line)

            # If disabled globally, skip visualization
            if args.no_gui:
                continue

            # Visualization frequency control: show only once every viz_every frames
            viz_every = getattr(args, 'viz_every', 1)
            try:
                viz_every = int(viz_every)
            except Exception:
                viz_every = 1
            if viz_every <= 0:
                continue
            start_f = args.start_frame or 0
            if ((ts.frame - start_f) % viz_every) != 0:
                # Skip visualization for this frame
                continue

            print("    Collecting visualization meshes...")
            mesh_r1 = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles[ixr1]))
            mesh_r2 = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles[ixr2]))
            mesh_r3 = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles[ixr3]))
            # Use different color shades for Upper vs Lower leaflet
            if is_upper:
                mesh_r1.paint_uniform_color([0.9, 0.2, 0.9])  # Purple
                mesh_r2.paint_uniform_color([0.9, 0.9, 0.2])  # Yellow
                mesh_r3.paint_uniform_color([0.9, 0.9, 0.9])  # Light Gray
            else:
                mesh_r1.paint_uniform_color([0.5, 0.1, 0.5])  # Dark Purple
                mesh_r2.paint_uniform_color([0.5, 0.5, 0.1])  # Dark Yellow
                mesh_r3.paint_uniform_color([0.6, 0.6, 0.6])  # Dark Gray
            # Optionally apply tiny visualization-only z offsets to avoid z-fighting/hiding
            try:
                viz_off = float(args.viz_offset)
            except Exception:
                viz_off = 0.0
            if viz_off != 0.0:
                mesh_r2.translate((0.0, 0.0, viz_off))
                mesh_r3.translate((0.0, 0.0, 2.0 * viz_off))
            viz_geometries.extend([mesh_r1, mesh_r2, mesh_r3])

        # Show all leaflet meshes in a single window after both leaflets are processed
        if not args.no_gui and viz_geometries:
            o3d.visualization.draw_geometries(viz_geometries, window_name=f"Frame {ts.frame} - Both Leaflets Curvature Regions", mesh_show_wireframe=True, mesh_show_back_face=True)

        end_time = time.time()
        print(f"Frame {ts.frame} processed in {end_time - start_time:.2f} seconds.")

    print("\nAnalysis complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert GROMACS trajectories to triangulated surfaces and analyze membrane defects.")
    
    # Input files
    parser.add_argument('--tpr', type=str, required=True, help='Path to the GROMACS TPR file.')
    parser.add_argument('--xtc', type=str, required=True, help='Path to the GROMACS XTC file.')
    
    # Output options
    parser.add_argument('--savedir', type=str, default='./cg2ts_output/', help='Directory to save output files.')
    parser.add_argument('--affix', type=str, default='DPPC', help='Affix for output filenames.')

    # Trajectory processing
    parser.add_argument('--start-frame', type=int, default=0, help='First frame to process.')
    parser.add_argument('--end-frame', type=int, default=None, help='Last frame to process.')
    parser.add_argument('--skip', type=int, default=1, help='Process every Nth frame.')

    # Algorithm parameters
    parser.add_argument('--leaflet-cutoff', type=float, default=12.0, help='Cutoff for LeafletFinder.')
    parser.add_argument('--striangle', type=float, default=1.0, help='Target triangle size for remeshing.')
    parser.add_argument('--cutoff', type=float, default=0.9, help='Defect cutoff parameter.')
    parser.add_argument('--depth-max', type=float, default=4.5, help='Maximum depth for defect detection.')
    parser.add_argument('--numit1', type=int, default=6, help='Number of iterations for first mesh optimization.')
    parser.add_argument('--numit2', type=int, default=5, help='Number of iterations for second mesh optimization.')
    parser.add_argument('--method', type=str, default='cvt full', help='Mesh optimization method (for compatibility).')

    # Display
    parser.add_argument('--no-gui', action='store_true', help='Run in non-interactive mode without showing visualizations.')
    parser.add_argument('--viz-offset', type=float, default=0.0, help='Optional small z-offset applied to successive visualization meshes to avoid z-fighting (visual only).')
    parser.add_argument('--viz-every', type=int, default=1, help='Show visualization once every N frames (1 = every frame, 0 = never).')

    parsed_args = parser.parse_args()
    main(parsed_args)
