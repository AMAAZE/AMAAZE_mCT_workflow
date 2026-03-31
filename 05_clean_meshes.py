#!/usr/bin/env python3

"""
05_clean_meshes.py

Original processing logic: RileyWilde
Refactoring and workflow design: Katrina E. Yezzi-Woodley

Post-process generated meshes by removing small disconnected components
and retaining the primary specimen geometry.

Cleaned meshes are written to the Clean_Meshes folder, ensuring more stable
and interpretable geometry for downstream analysis.
"""

from utils import *

meshsubfolder = os.path.normpath(os.path.join(scanpath, "Meshes"))
newmeshsubfolder = os.path.normpath(os.path.join(scanpath, "Clean_Meshes"))

if not os.path.isdir(meshsubfolder):
    raise RuntimeError(f"Mesh folder not found: {meshsubfolder}")

os.makedirs(newmeshsubfolder, exist_ok=True)

ddd = os.listdir(meshsubfolder)

fnames = []
for f in ddd:
    if f.lower().endswith('.ply'):
        fnames.append(f)

if len(fnames) == 0:
    raise RuntimeError(
        f"No .ply files found in {meshsubfolder}. "
        "Failure may have happened during surfacing. "
        "Run 04_surface.py first or check surfacing errors."
    )

print(f"Found {len(fnames)} .ply file(s) in {meshsubfolder}")
print(f"Cleaned meshes will be written to {newmeshsubfolder}")

num_cores = (
    clean_ncores
    if clean_ncores is not None
    else max(1, int(multiprocessing.cpu_count() / 4))
)

Parallel(n_jobs=num_cores)(
    delayed(clean_mesh_file)(f, meshsubfolder, newmeshsubfolder) for f in fnames
)

print("Mesh cleanup complete.")
print("Note: hole detection failed for some meshes; fallback cleanup was applied.")
       
   
