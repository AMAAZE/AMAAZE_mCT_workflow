#!/usr/bin/env python3

"""
02_build_volume.py

Original processing logic: RileyWilde
Refactoring and workflow design: Katrina E. Yezzi-Woodley

Load all CT slices, apply the rotation and cropping defined in controls.txt,
and construct a subsampled 3D volume by averaging slices in z-windows.

Each averaged slice is resized with aspect-ratio preservation,
using a maximum edge length of 225 pixels for computational efficiency.

The resulting volume is saved to ct<scan_num>_new.npz for downstream
segmentation and extraction.
"""

from utils import *

slicepath = os.path.normpath(slicepath)

if zwindow <= 0:
    raise RuntimeError("zwindow must be a positive integer.")
    
if not os.path.isdir(slicepath):
    raise RuntimeError("The provided slicepath does not exist or is not a folder.")
  
fnames = [
    f for f in os.listdir(slicepath)
    if f.lower().endswith((".tif", ".tiff"))
]

fnames_with_idx = [(f, extract_index(f)) for f in fnames]
fnames_with_idx.sort(key=lambda x: x[1])

fnames = [f for f, _ in fnames_with_idx]
indices = [idx for _, idx in fnames_with_idx]

n_slices = len(fnames)

if n_slices == 0:
    raise RuntimeError("No .tif or .tiff files found in the provided slicepath.")   
    
controls_fname = os.path.join(scanpath, "controls.txt")
    
if not os.path.exists(controls_fname):
    raise RuntimeError("controls.txt not found. Run 01_set_rotation.py first to generate it.")

with open(controls_fname, 'r') as file:
    text = file.read()    
    
q = text.split('\n')
    
angstr = 'ang2rot'
rowstr = 'rowrng'
colstr = 'colrng'
transposestr = 'transpose_preview'

for s in q:
    if s.startswith(angstr):
        ang2rot = literal_eval(s.split(':', 1)[1].strip())
    if s.startswith(rowstr): 
        rowrng = literal_eval(s.split(':', 1)[1].strip())
    if s.startswith(colstr):
        colrng = literal_eval(s.split(':', 1)[1].strip())
    if s.startswith(transposestr):
        transpose_preview = literal_eval(s.split(':', 1)[1].strip())
        
subsampled = []
for i in range(n_slices):
    #print(i/n_slices,fnames[i])
    im = io.imread(os.path.join(slicepath,fnames[i])).astype(int)
    # Do not use cv.imread here; it does not preserve the original voxel values. 
    # print(i/n_slices,fnames[i])#,im.max())
    if i%zwindow==0: #first
        imstack = im
    else: #middle:end
        imstack = imstack+im

    if i%zwindow==(zwindow-1): #last
        m = imstack.min()
        imstack = apply_preview_orientation(imstack, transpose_preview)
        imstack = rotate(imstack, ang2rot, preserve_range=True, resize=True, cval=m)
        print(i/n_slices, fnames[i], m)
        imstack = imstack[rowrng[0]:rowrng[1], colrng[0]:colrng[1]].copy() / zwindow
        
        imstack = resize_preserve_aspect(imstack, max_edge=225)
        
        subsampled.append(imstack)

rem = 0
if i%zwindow!=(zwindow-1): #fix the end if 'last' cond didn't happen
    imstack = apply_preview_orientation(imstack, transpose_preview)
    imstack = rotate(imstack, ang2rot, preserve_range=True, resize=True, cval=imstack.min())
    imstack = imstack[rowrng[0]:rowrng[1], colrng[0]:colrng[1]].copy() / ((i % zwindow) + 1)

    imstack = resize_preserve_aspect(imstack, max_edge=225)
    subsampled.append(imstack)
    rem = (i%zwindow) +1

outfile = os.path.join(scanpath, f"ct{scan_num}_new.npz")

# NOTE(dev):
# Reduced volume currently saved with np.savez() and default NumPy dtypes.
# File size and dtype optimization intentionally deferred until downstream (see utils)
# downstream geometric and segmentation behavior on rectangular datasets.
    
np.savez(
    outfile, 
    vol=subsampled, 
    rowrng=rowrng, 
    colrng=colrng, 
    ang=ang2rot, 
    origsz=im.shape, 
    remainder=rem,
    reduced_shape=np.array(subsampled[0].shape),
    transpose_preview=transpose_preview
)

# ============================================================
# Update run metadata
# ============================================================

run_metadata = load_run_metadata(scanpath, scan_num)

run_metadata["workflow"]["02_build_volume"] = {
    "input_slices": {
        "n_slices": n_slices,
        "first_slice": fnames[0],
        "last_slice": fnames[-1],
    },
    "processing": {
        "zwindow": zwindow,
        "remainder": rem,
        "ang2rot": ang2rot,
        "rowrng": rowrng,
        "colrng": colrng,
        "transpose_preview": transpose_preview,
    },
    "outputs": {
        "npz_file": outfile,
        "reduced_shape": list(np.array(subsampled).shape),
    },
}

save_run_metadata(scanpath, scan_num, run_metadata)



















