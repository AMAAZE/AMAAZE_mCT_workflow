#!/usr/bin/env python3

"""
01_set_rotation_crop.py

Original processing logic: RileyWilde
Refactoring and workflow design: Katrina E. Yezzi-Woodley

Guided visual setup step for the AMAAZE mCT surfacing workflow.

This script loads the metadata created by 00_share_data.py, opens
interactive preview windows for representative-slice orientation and crop
selection, records the selected rotation, row/column orientation, crop bounds,
and z-window, and updates the workflow metadata for downstream processing.
"""

# ============================================================
# Configuration and imports
# ============================================================

from utils import *

# ============================================================
# Load metadata
# ============================================================
timer_01_start = timeit.default_timer()

TOTAL_QUESTIONS_01 = 4

print_terminal_header("Step 2 of 5: Set Representative Slice Orientation")

print("In this step, you will prepare a representative slice")
print("so the workflow can build the correct working volume.")
print()
print("You will:")
print()
print("    1. Provide the appropriate filepath for the workflow")
print("    2. Orient the representative slice")
print("    3. Set the crop region")
print("    4. Choose the z-window for subvolume building")
print()
print("The first and last steps are completed by answering questions in the terminal.")
print()
print("Steps 2 and 3 are completed using interactive preview windows.")
print()
print("When you're ready, press Enter to begin.")
input("> ")

print_question_header("Workflow Metadata JSON", 1, TOTAL_QUESTIONS_01)

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

diagnostic_figures_path_01 = os.path.join(
    md.output_path,
    "01_diagnostic_figures"
)
os.makedirs(diagnostic_figures_path_01, exist_ok=True)

representative_slice_raw_png = os.path.join(
    diagnostic_figures_path_01,
    f"{md.dataset_folder_name}_representative_slice_raw.png"
)

plt.imsave(
    representative_slice_raw_png,
    raw_image,
    cmap="gray"
)

# ============================================================
# Set representative slice orientation
# ============================================================

rotation_angle, transpose_preview = choose_rotation_angle_interactively(
    raw_image,
    md.dataset_folder_name
)

oriented_image = apply_preview_orientation(
    raw_image,
    transpose_preview,
)

rotated_image = apply_preview_rotation(
    oriented_image,
    rotation_angle
)

representative_slice_oriented_png = os.path.join(
    diagnostic_figures_path_01,
    f"{md.dataset_folder_name}_representative_slice_oriented.png"
)

plt.imsave(
    representative_slice_oriented_png,
    rotated_image,
    cmap="gray"
)

# ============================================================
# Set crop
# ============================================================

rowrng, colrng = collect_crop_bounds_with_guides(
    rotated_image,
    md.dataset_folder_name
)

cropped_image = rotated_image[
    rowrng[0]:rowrng[1],
    colrng[0]:colrng[1]
].copy()

representative_slice_cropped_png = os.path.join(
    diagnostic_figures_path_01,
    f"{md.dataset_folder_name}_representative_slice_cropped.png"
)

plt.imsave(
    representative_slice_cropped_png,
    cropped_image,
    cmap="gray"
)

print()
print_success("Representative slice orientation accepted.")
print()
print_success("Crop region accepted.")
print()

# ============================================================
# Set z-window
# ============================================================

print_question_header("Reduced Working Volume", 4, 4)

print("The next workflow step builds a reduced working volume.")
print()
print("The z-window controls how many neighboring slices are averaged together.")
print("Higher values reduce noise, but may smooth fine detail.")
print()
print("Recommended default: 20")
print()

zwindow = ask(
    "Choose a z-window value.\n"
    "Press Enter to accept the recommended default.",
    default=20,
    cast=int
)

timer_01_stop = timeit.default_timer()

# ============================================================
# Calculate runtimes
# ============================================================

runtime_01_seconds = timer_01_stop - timer_01_start

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
        
        "diagnostic_figures_path_01": diagnostic_figures_path_01,
        "representative_slice_raw_png": representative_slice_raw_png,
        "representative_slice_oriented_png": representative_slice_oriented_png,
        "representative_slice_cropped_png": representative_slice_cropped_png,
        
        "zwindow": zwindow,
}

save_metadata(metadata_path, metadata)

# ============================================================
# Confirm completion
# ============================================================

print_step_complete_header("Step 2 Complete")

print("Representative slice setup complete.")
print()
print(f"Setup took {format_runtime(runtime_01_seconds)} to complete.")
print()
print("Metadata updated:")
print()
print(f"    {metadata_path}")
print()
print("Important: Use this metadata JSON to continue the workflow later.")
print()

ask_run_next_step("02_build_subvolume.py", metadata_path)

