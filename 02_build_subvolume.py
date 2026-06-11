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

timer_02_start = timeit.default_timer()

print()
dataset_path = ask_existing_path(
    "What is the full path to the dataset folder you want to continue working on?\n"
    "Please include the dataset folder itself in the path.\n"
    "This should be the same dataset folder path you gave to 00_share_data.py.\n"
    "Example:\n"
    "C:/MyProject/CT_scan_01",
    is_dir=True
)

metadata_path = find_metadata_file_in_dataset(dataset_path)
metadata = load_metadata_if_available(metadata_path)

scanpath = metadata["00_share_data"]["scanpath"]
slicepath = metadata["00_share_data"]["slicepath"]
output_path = metadata["00_share_data"]["output_path"]
slice_index_fraction = metadata["00_share_data"]["slice_index_fraction"]

dataset_folder_name = metadata["00_share_data"]["dataset_folder_name"]

transpose_preview = metadata["01_set_rotation_crop"]["transpose_preview"]
rotation_angle = metadata["01_set_rotation_crop"]["rotation_angle"]

rowrng = metadata["01_set_rotation_crop"]["rowrng"]
colrng = metadata["01_set_rotation_crop"]["colrng"]

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

subvolume_file = os.path.join(
    output_path,
    f"{dataset_folder_name}_subvolume.npz"
)

# NOTE(dev):
# Reduced volume currently saved with np.savez() and default NumPy dtypes.
# File size and dtype optimization intentionally deferred until downstream (see utils)
# downstream geometric and segmentation behavior on rectangular datasets.
    
np.savez(
    subvolume_file, 
    vol=subsampled, 
    rowrng=rowrng, 
    colrng=colrng, 
    ang=rotation_angle, 
    origsz=im.shape, 
    remainder=rem,
    entire_subvolume_shape=np.array(subsampled).shape,
    subvolume_slice_shape=np.array(subsampled[0].shape),
    transpose_preview=transpose_preview
)

entire_subvolume_shape = list(np.array(subsampled).shape)
subvolume_slice_shape = list(np.array(subsampled[0]).shape)

timer_02_stop = timeit.default_timer()

# ============================================================
# Calculate runtimes
# ============================================================

runtime_02_seconds = timer_02_stop - timer_02_start
print("02_build_subvolume.py runtime: ", runtime_02_seconds)

# ============================================================
# Update metadata
# ============================================================

metadata["02_build_subvolume"] = {
    "status": "complete",
    
    "zwindow": zwindow,
    "remainder": rem,
    
    "subvolume_file": subvolume_file,
    "entire_subvolume_shape": entire_subvolume_shape,
    "subvolume_slice_shape": subvolume_slice_shape,
    
    "runtime_seconds": runtime_02_seconds,
}

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















