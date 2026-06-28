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

metadata_paths = get_metadata_paths_from_command_line_or_user(
    step_name="01_set_rotation_crop",
    allow_batch=False
)

metadata_path = metadata_paths[0]
metadata = load_metadata_if_available(metadata_path)

md = unpack_metadata(metadata)

# ============================================================
# Load slices and identify representative slice
# ============================================================

slice_files, slice_indices = get_sorted_slice_files(md.slicepath)

slice_index = int(len(slice_files) * md.slice_index_fraction)
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
    md.dataset_folder_name
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
    md.dataset_folder_name
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
    md.dataset_folder_name
)

rotated_image = apply_preview_rotation(
    oriented_image,
    rotation_angle
)


# ============================================================
# Set crop
# ============================================================

while True:

    rowrng, colrng = collect_crop_bounds(rotated_image, md.dataset_folder_name)
    
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
        f"{md.dataset_folder_name} \n"
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

# ============================================================
# Set z-window
# ============================================================

zwindow = ask(
    "zwindow controls how many adjacent slices are averaged when building the reduced working volume.\n"
    "Use 1 to keep every slice.",
    default=20,
    cast=int
) 

if zwindow <= 0:
    print()
    print("zwindow must be a positive integer.")
    print("Please rerun this step and choose 1 or higher.")
    raise SystemExit

timer_01_stop = timeit.default_timer()
# ============================================================
# Calculate runtimes
# ============================================================

runtime_01_seconds = timer_01_stop - timer_01_start
print("01_set_crop_rotation.py runtime: ", runtime_01_seconds)

# ============================================================
# Update metadata
# ============================================================

metadata.setdefault("workflow_runtimes", {})
metadata["workflow_runtimes"]["runtime_01_seconds"] = runtime_01_seconds

metadata["01_set_rotation_crop"] = {
        "status": "complete",
    
        "transpose_preview": transpose_preview,
        "rotation_angle": rotation_angle, 
        "rowrng": rowrng,
        "colrng": colrng, 
        
        "zwindow": zwindow,
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

