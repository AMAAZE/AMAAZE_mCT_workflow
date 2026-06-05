#!/usr/bin/env python3

"""
00_share_data.py

Original source code: Katrina E. Yezzi-Woodley

Guided intake step for the AMAAZE mCT surfacing workflow.

This script interviews the user, checks the scan folder, slice folder,
and layout CSV, confirms that supported slice files can be found and sorted,
creates an output folder, and writes the first metadata file for the workflow.
"""

from utils import *

# ============================================================
# INTERVIEW USER ABOUT INPUT DATA
# ============================================================

print()
print("Welcome to the AMAAZE mCT surfacing workflow.")
print("We are glad you are here.")
print()
print("Before we begin, we will ask a few questions about your data.")
print()

dataset_name = ask(
    "What short name should we use for this dataset?\n"
    "Example: bonefrags, teeth, beads"
)  

dataset_name = sanitize_name(dataset_name)

print()
scan_num = ask(
    "We index scans by number because some datasets include a series of scans.\n"
    "What scan number should we use for this dataset?\n"
    "The default is 1. You can accept the default by pressing Enter.",
    default=1,
    cast=int
)

print()
scanpath = ask_existing_path(
    "What is the path to the folder for this scan dataset?\n"
    "This is the main data folder for the scan we are processing.\n"
    "Please include the dataset folder itself in the path.\n"
    "Example:\n"
    "C:/MyProject/CT_scan_01",
    is_dir=True
)

output_path = create_output_folder(scanpath, dataset_name, scan_num)

print()
print(f"Workflow outputs will be saved here:")
print(output_path)
print()

slicepath = ask_existing_path(
    "What is the full path to the folder containing the slice files?\n"
    "Please include the slice folder itself in the path.\n"
    "Example:\n"
    "C:/MyProject/CT_scan_01/Slices",
    is_dir=True
)

slice_files, slice_indices = get_sorted_slice_files(slicepath)

print()
print(f"Found {len(slice_files)} supported slice files.")
print("First 5 slices:", [os.path.basename(f) for f in slice_files[:5]])
print("Last 5 slices:", [os.path.basename(f) for f in slice_files[-5:]])
print()
print("Slice filenames were sorted by the numeric index in each filename.")
print("Skipped numbers are okay. Duplicate numeric indices are not okay.")
print()

layoutfile = ask_existing_path(
    "What is the path to the layout CSV file?\n"
    "Please include the filename in the path.\n"
    "This file tells the workflow which specimens are expected in the scan and where they are located within the scan.",
    is_dir=False
)

print()
slice_index_fraction = ask_float_in_range(
    "Choose a representative slice fraction for previewing in the next step.\n"
    "This should land near the middle of a tier that contains specimens.\n"
    "For one occupied tier, 0.50 is often good. For two occupied tiers, 0.25 or 0.75 may be better.\n"
    "Avoid choosing a value that falls within an empty tier.\n"
    "If the scan appears upside down relative to the layout later, we will correct that later in the workflow.\n"
    "The default is 0.50. Press Enter to accept the default, or type a different value.",
    minimum=0.0,
    maximum=1.0,
    default=0.50
)

print()
print("Voxel information controls the real-world scale of your meshes.")
print("If your scan is isotropic, voxel size and slice spacing are the same.")
print()

voxel_size_mm = ask(
    "What is the in-plane voxel size in mm?",
    cast=float
)

print()
is_isotropic = ask_yes_no(
    "Is the scan isotropic, meaning slice spacing equals in-plane voxel size?",
    default="y"
)

if is_isotropic:
    voxel_spacing_mm = None
else:
    voxel_spacing_mm = ask(
        "What is the slice spacing in mm?",
        cast=float
    )


# ============================================================
# CREATE INITIAL WORKFLOW METADATA
# ============================================================

metadata_filename = build_metadata_filename(dataset_name, scan_num)
metadata_path = os.path.join(output_path, metadata_filename)

metadata = {
    "dataset_name": dataset_name,
    "scan_num": scan_num,

    "paths": {
        "scanpath": scanpath,
        "slicepath": slicepath,
        "layoutfile": layoutfile,
        "output_path": output_path,
        "metadata_path": metadata_path
    },

    "input_slices": {
        "supported_extensions": list(SUPPORTED_SLICE_EXTENSIONS),
        "n_slices": len(slice_files),
        "first_slice": os.path.basename(slice_files[0]),
        "last_slice": os.path.basename(slice_files[-1]),
        "first_slice_index": int(slice_indices[0]),
        "last_slice_index": int(slice_indices[-1]),
        "slice_indices_are_consecutive": slice_indices == list(range(slice_indices[0], slice_indices[0] + len(slice_indices)))
    },

    "user_choices": {
        "slice_index_fraction": slice_index_fraction,
        "voxel_size_mm": voxel_size_mm,
        "voxel_spacing_mm": voxel_spacing_mm,
    },

    "workflow": {
        "00_share_data": {
            "status": "complete"
        }
    },

    "outputs": {},
    "warnings": []
}

save_metadata(metadata_path, metadata)

print()
print("Setup complete.")
print(f"Metadata saved to:")
print(metadata_path)
print()
print("Next step:")
print("python 01_set_rotation_crop.py")
