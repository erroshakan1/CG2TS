# CG2TS v4 — Membrane Defect Analysis Tool

Coarse-Grained to Triangulated Surface (CG2TS) extracts and analyzes membrane packing defects from molecular dynamics trajectories.

Quick start
-----------
1. Install dependencies (use a virtual environment):

```bash
pip install -r requirements.txt
```

2. Run in headless mode:

```bash
python CG2TS_v4.py --tpr md.tpr --xtc md.xtc --savedir ./results --no-gui
```

Notes
-----
- Default cutoff for leaflet separation: 15.0 Å. This value works well for most room-temperature fluid-phase systems. If the two leaflets appear merged or overlapping in visualizations, reduce the `--leaflet-cutoff` (for example, try 12 Å):

```bash
python CG2TS_v4.py --tpr md.tpr --xtc md.xtc --leaflet-cutoff 12 --savedir ./results
```

Ubuntu / Qt GUI errors
----------------------
On some Ubuntu systems running PyQt/Qt-based viewers you may see errors about the Qt platform plugin or XDG session type. Set these environment variables before launching the GUI to avoid the errors:

```bash
export QT_QPA_PLATFORM=xcb
export XDG_SESSION_TYPE=x11
```

Contact
-------
Author: h.camoglu@outlook.com
>>>>>>> 8ad2e6b (Initial import: selected files for CG2TS_v4)
