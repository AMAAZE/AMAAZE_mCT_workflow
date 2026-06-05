#!/usr/bin/env python3

"""
02_build_volume.py

Original processing logic: RileyWilde
Refactoring and workflow design: Katrina E. Yezzi-Woodley

Load all CT slices, apply the rotation and cropping defined in previous step,
and construct a subsampled 3D volume by averaging slices in z-windows.

Each averaged slice is resized with aspect-ratio preservation,
using a maximum edge length of 225 pixels for computational efficiency.

The resulting volume is saved to an .npz and the metadata file is updated
for downstream segmentation and extraction.
"""

# ============================================================
# Configuration and imports
# ============================================================

from utils import *

# ============================================================
# Load metadata
# ============================================================

print()
dataset_path = ask_existing_path(
    "What is the name of the dataset folder you want to continue working on?\n"
    "This should be the same dataset folder you gave to 00_share_data.py.",
    is_dir=True
)

metadata_path = find_metadata_file_in_dataset(dataset_path)
metadata = load_metadata_if_available(metadata_path)

scanpath = metadata["paths"]["scanpath"]
slicepath = metadata["paths"]["slicepath"]
output_path = metadata["paths"]["output_path"]
slice_index_fraction = metadata["user_choices"]["slice_index_fraction"]

dataset_name = metadata["dataset_name"]
scan_num = metadata["scan_num"]

transpose_preview = metadata["orientation"]["transpose_preview"]
rotation_angle = metadata["orientation"]["rotation_angle"]

rowrng = metadata["cropping"]["rowrng"]
colrng = metadata["cropping"]["colrng"]

# ============================================================
# Load slices
# ============================================================

slice_files, slice_indices = get_sorted_slice_files(slicepath)

# ============================================================
# Set z-window
# ============================================================

zwindow = ask(
    "zwindow controls how many adjacent slices are averaged when building the reduced working volume.\n"
    "Use 1 to keep every slice.",
    default=1,
    cast=int
) 

if zwindow <= 0:
    print()
    print("zwindow must be a positive integer.")
    print("Please rerun this step and choose 1 or higher.")
    raise SystemExit

# ============================================================
# build subvolume
# ============================================================

n_slices = len(slice_files)

subsampled = []

for i, slice_file in enumerate(slice_files):
    #print(i/n_slices,fnames[i])
    im = read_slice(slice_file).astype(int)
    # Do not use cv.imread here; it does not preserve the original voxel values. 
    # print(i/n_slices,fnames[i])#,im.max())
    if i%zwindow==0: #first
        imstack = im
    else: #middle:end
        imstack = imstack+im

    if i%zwindow==(zwindow-1): #last
        m = imstack.min()
        imstack = apply_preview_orientation(imstack, transpose_preview)
        imstack = rotate(imstack, rotation_angle, preserve_range=True, resize=True, cval=m)
        print(i / n_slices, os.path.basename(slice_file), m)
        imstack = imstack[rowrng[0]:rowrng[1], colrng[0]:colrng[1]].copy() / zwindow
        
        imstack = resize_preserve_aspect(imstack, max_edge=225)
        
        subsampled.append(imstack)

rem = 0
if i%zwindow!=(zwindow-1): #fix the end if 'last' cond didn't happen
    imstack = apply_preview_orientation(imstack, transpose_preview)
    imstack = rotate(imstack, rotation_angle, preserve_range=True, resize=True, cval=imstack.min())
    imstack = imstack[rowrng[0]:rowrng[1], colrng[0]:colrng[1]].copy() / ((i % zwindow) + 1)

    imstack = resize_preserve_aspect(imstack, max_edge=225)
    subsampled.append(imstack)
    rem = (i%zwindow) +1

outfile = os.path.join(
    output_path,
    f"{dataset_name}_scan{scan_num}_subvolume.npz"
)

# NOTE(dev):
# Reduced volume currently saved with np.savez() and default NumPy dtypes.
# File size and dtype optimization intentionally deferred until downstream (see utils)
# downstream geometric and segmentation behavior on rectangular datasets.
    
np.savez(
    outfile, 
    vol=subsampled, 
    rowrng=rowrng, 
    colrng=colrng, 
    ang=rotation_angle, 
    origsz=im.shape, 
    remainder=rem,
    reduced_shape=np.array(subsampled[0].shape),
    transpose_preview=transpose_preview
)

# ============================================================
# Update metadata
# ============================================================

metadata["workflow"]["02_build_volume"] = {
    "status": "complete",
    "input_slices": {
        "n_slices": n_slices,
        "first_slice": os.path.basename(slice_files[0]),
        "last_slice": os.path.basename(slice_files[-1]),
    },
    "processing": {
        "zwindow": zwindow,
        "remainder": rem,
        "rotation_angle": rotation_angle,
        "rowrng": rowrng,
        "colrng": colrng,
        "transpose_preview": transpose_preview,
    },
    "outputs": {
        "npz_file": outfile,
        "reduced_shape": list(np.array(subsampled).shape),
    },
}

metadata["outputs"]["npz_file"] = outfile

save_metadata(metadata_path, metadata)

# ============================================================
# Confirm completion
# ============================================================
print()
print("Subvolume created.")
print("Metadata updated:")
print(metadata_path)
print()
print("Next step:")
print("python 03_segment.py")















