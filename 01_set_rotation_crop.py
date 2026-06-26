#!/usr/bin/env python3

"""
01_set_rotation_crop.py

Original processing logic: RileyWilde
Refactoring and workflow design: Katrina E. Yezzi-Woodley

Preview a representative CT slice using the current user settings,
apply preview orientation, rotation, and crop, and write the selected
rotation/crop settings to controls.txt for downstream processing.
"""

# ============================================================
# Configuration and imports
# ============================================================

from utils import *

# ============================================================
# Load metadata
# ============================================================
timer_01_start = timeit.default_timer()

#print()
#dataset_path = ask_existing_path(
#    "What is the full path to the dataset folder you want to continue working on?\n"
#    "This should be the same dataset folder path you gave to 00_share_data.py.\n"
#    "Example:\n"
#    "C:/MyProject/CT_scan_01",
#    is_dir=True
#)
#
#metadata_path = find_metadata_file_in_dataset(dataset_path)
#metadata = load_metadata_if_available(metadata_path)

metadata_paths = get_metadata_paths_from_command_line_or_user(
    step_name="01_set_rotation_crop",
    allow_batch=False
)

metadata_path = metadata_paths[0]
metadata = load_metadata_if_available(metadata_path)

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
# Load slices and identify representative slice
# ============================================================

slice_files, slice_indices = get_sorted_slice_files(slicepath)

slice_index = int(len(slice_files) * slice_index_fraction)
slice_index = min(slice_index, len(slice_files) - 1)

slice_file = slice_files[slice_index]
raw_image = read_slice(slice_file)

print()
print("Using representative slice:")
print(os.path.basename(slice_file))

# ============================================================
# Set transpose
# ============================================================
transpose_preview = False
oriented_image = apply_preview_orientation(raw_image, transpose_preview)

fig, ax = plt.subplots()
plt.show(block=False)
update_preview(
    ax, 
    fig, 
    oriented_image, 
    f"Representative slice | transpose_preview={transpose_preview}",
    dataset_folder_name
)

while True:

    print()
    print("Compare the preview window to your layout CSV.")
    print("If rows and columns appear swapped relative to the layout, transpose the preview.")
    print()

    transpose_preview = ask_yes_no(
        "Would you like to transpose the preview image?",
        default="n"
    )

    oriented_image = apply_preview_orientation(raw_image, transpose_preview)

    update_preview(ax, 
        fig, 
        oriented_image, 
        f"Representative slice | transpose_preview={transpose_preview}",
    dataset_folder_name
    )

    satisfied = ask_yes_no(
        "After transposing, do the rows and columns appear correct relative to the layout CSV?",
        default="y"
    )

    if satisfied:
        break

    plt.close(fig)

# ============================================================
# Set rotation (New interactive version)
# ============================================================

rotation_angle = choose_rotation_angle_interactively(
    oriented_image,
    dataset_folder_name
)

rotated_image = apply_preview_rotation(
    oriented_image,
    rotation_angle
)


# ============================================================
# Set crop
# ============================================================

while True:

    rowrng, colrng = collect_crop_bounds(rotated_image, dataset_folder_name)
    
    if rowrng is None or colrng is None:
        continue

    cropped_image = rotated_image[
        rowrng[0]:rowrng[1],
        colrng[0]:colrng[1]
    ].copy()

    print("Rotated image shape:", rotated_image.shape)
    print("Cropped image shape:", cropped_image.shape)

    plt.figure()
    plt.imshow(cropped_image, cmap="gray")
    plt.title(
        f"{dataset_folder_name} \n"
        f"Crop preview | rows={rowrng}, cols={colrng}"
    )
    plt.axis("off")
    plt.show(block=False)

    satisfied = ask_yes_no(
        "Does this crop look correct?",
        default="y"
    )

    if satisfied:
        break
    else:
        plt.close()
        print()
        print("Let's try the crop again.")

timer_01_stop = timeit.default_timer()

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
# Calculate runtimes
# ============================================================

runtime_01_seconds = timer_01_stop - timer_01_start
print("01_set_crop_rotation.py runtime: ", runtime_01_seconds)

# ============================================================
# Update metadata
# ============================================================

metadata["01_set_rotation_crop"] = {
        "status": "complete",
    
        "transpose_preview": transpose_preview,
        "rotation_angle": rotation_angle, 
        "rowrng": rowrng,
        "colrng": colrng, 
    
        "runtime_01_seconds": runtime_01_seconds,
}

save_metadata(metadata_path, metadata)


# ============================================================
# Confirm completion
# ============================================================
print()
print("Rotation and crop setup complete.")
print("Metadata updated:")
print(metadata_path)
print()

ask_run_next_step("02_build_subvolume.py", metadata_path)

