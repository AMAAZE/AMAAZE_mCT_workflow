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

#print()
#dataset_path = ask_existing_path(
#    "What is the full path to the dataset folder you want to continue working on?\n"
#    "This should be the same dataset folder path you gave to 00_share_data.py.\n"
#    "Example:\n"
#    "C:/MyProject/CT_scan_01",
#    is_dir=True
#)
#metadata_path = find_metadata_file_in_dataset(dataset_path)
#metadata = load_metadata_if_available(metadata_path)


metadata_paths = get_metadata_paths_from_command_line_or_user(
    step_name="02_build_subvolume",
    allow_batch=True
)

#metadata_path = metadata_paths[0]
#metadata = load_metadata_if_available(metadata_path)

for metadata_path in metadata_paths:

    timer_02_start = timeit.default_timer()
    
    metadata = load_metadata_if_available(metadata_path)

    print()
    print("Building subvolume for:")
    print(metadata_path)
    print()
    
    (
        dataset_folder_name,
        scanpath,
        slicepath,
        layoutfile,
        output_path,
        metadata_path,
        slice_index_fraction,
        voxel_size_mm,
        voxel_spacing_mm,
        transpose_preview,
        rotation_angle,
        rowrng,
        colrng,
        subvolume_file,
    ) = unpack_metadata(metadata)


# ============================================================
# Load slices
# ============================================================

    slice_files, slice_indices = get_sorted_slice_files(slicepath)


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

    metadata.setdefault("workflow_runtimes", {})
    metadata["workflow_runtimes"]["runtime_02_seconds"] = runtime_02_seconds

    metadata["02_build_subvolume"] = {
        "status": "complete",

        "resize_max_edge": 225,
        
        "remainder": rem,
    
        "subvolume_file": subvolume_file,
        "entire_subvolume_shape": entire_subvolume_shape,
        "subvolume_slice_shape": subvolume_slice_shape,
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

    if len(metadata_paths) == 1:
        ask_run_next_step("03_segment.py", metadata_path)
    else:
        print()
        print("Batch subvolume building complete.")
        print("To continue, run:")
        print("    python 03_segment.py")
        print("individually for each workflow that needs interactive segmentation.")
        print()














