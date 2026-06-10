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

print()
dataset_path = ask_existing_path(
    "What is the name of the dataset folder you want to continue working on?\n"
    "This should be the same dataset folder you gave to 00_share_data.py.",
    is_dir=True
)

metadata_path = find_metadata_file_in_dataset(dataset_path)
metadata = load_metadata_if_available(metadata_path)

scanpath = metadata["00_share_data"]["scanpath"]
slicepath = metadata["00_share_data"]["slicepath"]
output_path = metadata["00_share_data"]["output_path"]
slice_index_fraction = metadata["00_share_data"]["slice_index_fraction"]

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
update_preview(ax, fig, oriented_image, f"Representative slice | transpose_preview={transpose_preview}")

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

    update_preview(ax, fig, oriented_image, f"Representative slice | transpose_preview={transpose_preview}")

    satisfied = ask_yes_no(
        "Does this orientation look correct relative to the layout CSV?",
        default="y"
    )

    if satisfied:
        break


# ============================================================
# Set rotation
# ============================================================

rotation_angle = 0.0

while True:

    print()
    print("The next step is rotation.")
    print("Positive values rotate counterclockwise.")
    print("Negative values rotate clockwise.")
    print()

    rotation_angle = ask(
        "Enter a rotation angle in degrees.",
        default=rotation_angle,
        cast=float
    )

    rotated_image = apply_preview_rotation(
        oriented_image,
        rotation_angle
    )

    update_preview(ax, fig, rotated_image, f"Rotation = {rotation_angle} degrees")

    satisfied = ask_yes_no(
        "Does this rotation look correct?",
        default="y"
    )

    if satisfied:
        break

print()
print("Rotation accepted.")
print("Please close the rotation preview window to continue to cropping.")
input("Press Enter after closing the rotation preview window...")

plt.close(fig)

# ============================================================
# Set crop
# ============================================================

while True:

    rowrng, colrng = collect_crop_bounds(rotated_image)
    
    if rowrng is None or colrng is None:
        continue

    cropped_image = rotated_image[
        rowrng[0]:rowrng[1],
        colrng[0]:colrng[1]
    ].copy()

    plt.figure()
    plt.imshow(cropped_image, cmap="gray")
    plt.title(f"Crop preview | rows={rowrng}, cols={colrng}")
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
    
        "runtime_seconds": runtime_01_seconds,
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
print("Next step:")
print("python 02_build_subvolume.py")

