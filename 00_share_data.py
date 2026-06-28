#!/usr/bin/env python3

"""
00_share_data.py

Original source code: Katrina E. Yezzi-Woodley

Guided intake step for the AMAAZE mCT surfacing workflow.

This script interviews the user, checks the scan folder, slice folder,
and layout CSV, confirms that supported slice files can be found and sorted,
creates an output folder, and writes the first iteration of the metadata file for the workflow.
"""

# ============================================================
# Configuration and imports
# ============================================================

from utils import *

# ============================================================
# Welcome to the workflow
# ============================================================

timer_00_start = timeit.default_timer()

timer_00_start = timeit.default_timer()

TOTAL_QUESTIONS_00 = 7

print_terminal_header("Welcome to the AMAAZE mCT Surfacing Workflow")

print("This workflow consists of five steps:")
print()
print("    1 Share Data")
print("    2 Set Rotation & Crop")
print("    3 Build Subvolume")
print("    4 Segment")
print("    5 Surface")
print()
print("You are currently completing Step 1.")
print()
print("When you're ready, press Enter to begin.")

input("> ")

# ============================================================
# Request folder name
# ============================================================

print_question_header("Dataset Folder Name", 1, TOTAL_QUESTIONS_00)

dataset_folder_name = ask_dataset_folder_name(
    "The dataset folder name is used to name workflow outputs.\n"
    "Note: If it is very long, your workflow output names will also be very long.\n\n"
    "Example:\n\n"
    "    CT_scan_01\n\n"
    "What is the name of the dataset folder we will be processing today?"
)

print_success(f"Dataset folder name recorded: {dataset_folder_name}")

# ============================================================
# Request data filepath
# ============================================================

print_question_header("Dataset Filepath", 2, TOTAL_QUESTIONS_00)

scanpath = ask_existing_path(
    "What is the entire path to the folder for this scan dataset?\n"
    "Please include the dataset folder name itself in the path.\n\n"
    "Example:\n\n"
    "    C:/MyProject/CT_scan_01",
    is_dir=True
)

print_success("Dataset folder found.")

# ============================================================
# Slice filepath request and verification
# ============================================================

print_question_header("Slice Filepath", 3, TOTAL_QUESTIONS_00)

slicepath = ask_existing_path(
    "What is the full path to the folder containing the slice files?\n"
    "Please include the slice folder itself in the path.\n\n"
    "Example:\n\n"
    "    C:/MyProject/CT_scan_01/Slices",
    is_dir=True
)

print_success("Slice folder found.")

supported_extensions = list(SUPPORTED_SLICE_EXTENSIONS)

while True:

    try:
        slice_files, slice_indices = get_sorted_slice_files(slicepath)

    except RuntimeError as e:

        print()
        print(str(e))
        print()

        if "Duplicate numeric slice indices" in str(e):
            print()
            print("Each slice must have a unique numeric index.")
            print("Please fix the slice filenames and rerun 00_share_data.py.")
            raise SystemExit

        print()
        print("The slice folder could not be used.")
        print("Please check the folder path and slice files, then try again.")
        print()

        print_question_header("Slice Filepath", 3, TOTAL_QUESTIONS_00)

        slicepath = ask_existing_path(
            "What is the full path to the folder containing the slice files?",
            is_dir=True
        )

        continue

    n_slices = len(slice_files)
    total_slice_bytes = sum(os.path.getsize(f) for f in slice_files)
    total_slice_gb = total_slice_bytes / (1024 ** 3)

    first_slice = os.path.basename(slice_files[0])
    last_slice = os.path.basename(slice_files[-1])

    first_slice_index = int(slice_indices[0])
    last_slice_index = int(slice_indices[-1])

    slice_indices_are_consecutive = bool(
        (np.diff(slice_indices) == 1).all()
    )

    print_question_header("Verify Slices", 4, TOTAL_QUESTIONS_00)

    print(f"We found {n_slices} supported slice files.")
    print()
    print("First 5 slices:")
    print([os.path.basename(f) for f in slice_files[:5]])
    print()
    print("Last 5 slices:")
    print([os.path.basename(f) for f in slice_files[-5:]])
    print()
    print("Slice filenames were sorted by the numeric index in each filename.")

    if slice_indices_are_consecutive:
        print("Slice numbering is consecutive.")
    else:
        print("Skipped numeric indices were detected.")
        print("This is okay and will not affect processing.")

    print()

    if ask_yes_no(
        "Do these look like the correct slice files?",
        default="y"
    ):
        print_success("Slice files verified.")
        break

    print()
    print("Let's try a different slice folder.")
    print()

    print_question_header("Slice Filepath", 3, TOTAL_QUESTIONS_00)

    slicepath = ask_existing_path(
        "What is the full path to the folder containing the slice files?",
        is_dir=True
    )

# ============================================================
# Request CSV filepath
# ============================================================

print_question_header("CSV Filepath", 5, TOTAL_QUESTIONS_00)

layoutfile = ask_existing_path(
    "The layout CSV tells the workflow which specimens\n"
    "are expected in the scan and where they are located within the scan.\n\n"
    "The layout CSV should describe this dataset only and contain:\n\n"
    "    Column 1: Tier number\n"
    "    Column 2: Row number\n"
    "    Column 3+: Specimen identifiers\n\n"
    "What is the path to the layout CSV file?\n"
    "Please include the filename in the path.",
    is_dir=False
)

layout_filename = os.path.basename(layoutfile)

print_success("Layout CSV found.")

# ============================================================
# Request voxel information
# ============================================================

print_question_header("Voxel Size", 6, TOTAL_QUESTIONS_00)

print("Voxel information controls the real-world scale of your meshes.")
print("If your scan is isotropic, voxel size and slice spacing are the same.")
print()

voxel_size_mm = ask(
    "What is the in-plane voxel size in mm?",
    cast=float
)

is_isotropic = ask_yes_no(
    "Is the scan isotropic, meaning slice spacing equals in-plane voxel size?",
    default="y"
)

if is_isotropic:
    voxel_spacing_mm = None
    print_success("Voxel size recorded. Slice spacing will use the same value.")
else:
    voxel_spacing_mm = ask(
        "What is the slice spacing in mm?",
        cast=float
    )
    print_success("Voxel size and slice spacing recorded.")

output_path = create_output_folder(scanpath, dataset_folder_name)

print()
print("All workflow outputs will be saved here:")
print()
print(f"    {output_path}")
print()

# ============================================================
# Select representative slice
# ============================================================

print_question_header("Representative Slice", 7, TOTAL_QUESTIONS_00)

slice_index_fraction = ask_float_in_range(
    "The next step in the workflow will ask you to compare\n"
    "a representative slice to the layout CSV,\n"
    "and then based on what you see, select a few settings.\n\n"
    "The representative slice should land near the middle of a tier that contains specimens.\n\n"
    "    For one occupied tier, 0.50 is often good.\n"
    "    For two occupied tiers, 0.25 or 0.75 may be better.\n"
    "    Avoid choosing a value that falls within an empty tier.\n\n"
    "Note: If the scan appears upside down relative to the layout later,\n"
    "we will correct that later in the workflow.\n\n"
    "Choose a representative slice.\n\n"
    "Default: 0.50\n\n"
    "Press Enter to accept the default,\n"
    "or type a different value.",
    minimum=0.0,
    maximum=1.0,
    default=0.50
)

print_success(f"Representative slice fraction recorded: {slice_index_fraction}")

timer_00_stop = timeit.default_timer()

# ============================================================
# Calculate runtimes
# ============================================================

runtime_00_seconds = timer_00_stop - timer_00_start

# ============================================================
# Create metadata file and dictionaries
# ============================================================

metadata_filename = build_metadata_filename(dataset_folder_name)
metadata_path = os.path.join(output_path, metadata_filename)  

metadata= {}

# ============================================================
# Update metadata
# ============================================================

metadata["workflow_runtimes"] = {
    "runtime_00_seconds": runtime_00_seconds,
}

metadata["00_share_data"] = {
    "status": "complete",

    "data_locations": {
        "dataset_folder_name": dataset_folder_name, 
        "scanpath": scanpath,
        "slicepath": slicepath,
        "layout_filename": layout_filename,
        "layoutfile": layoutfile,
        "output_path": output_path,
        "metadata_path": metadata_path,
    },

    "slice_inventory": {
        "first_slice": first_slice,
        "first_slice_index": first_slice_index,
        "last_slice": last_slice,
        "last_slice_index": last_slice_index,
        "n_slices": n_slices,
        "slice_index_fraction": slice_index_fraction,
        "slice_indices_are_consecutive": slice_indices_are_consecutive,
        "supported_extensions": supported_extensions,
        "total_slice_bytes": total_slice_bytes,
        "total_slice_gb": total_slice_gb,
    },

    "voxel_information": {
        "is_isotropic": is_isotropic,
        "voxel_size_mm": voxel_size_mm,
        "voxel_spacing_mm": voxel_spacing_mm,
    },
}

save_metadata(metadata_path, metadata)


# ============================================================
# Confirm completion
# ============================================================

print_step_complete_header("Step 1 Complete")

print("Thank you for sharing the data.")
print()
print("Setup complete.")
print()
print(f"Setup took {format_runtime(runtime_00_seconds)} to complete.")
print()
print("Metadata saved to:")
print()
print(f"    {metadata_path}")
print()
print("Important: Use this metadata JSON to continue the workflow later.")
print()

# ============================================================
# Initiate next step
# ============================================================

ask_run_next_step("01_set_rotation_crop.py", metadata_path)




