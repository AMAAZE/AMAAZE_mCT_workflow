#!/usr/bin/env python3

"""
00_share_data.py

Original source code: Katrina E. Yezzi-Woodley

Guided intake step for the AMAAZE mCT surfacing workflow.

This script interviews the user, checks the scan folder, slice folder,
and layout CSV, confirms that supported slice files can be found and sorted,
creates an output folder, and writes the first metadata file for the workflow.
"""

# ============================================================
# Configuration and imports
# ============================================================

from utils import *

# ============================================================
# Interview user about input data
# ============================================================
timer_00_start = timeit.default_timer()

print()
print("Welcome to the AMAAZE mCT surfacing workflow.")
print("We are glad you are here.")
print()
print("Before we begin, we will ask a few questions about your data.")
print()

print()
dataset_folder_name = ask_dataset_folder_name(
    "What is the name of the folder where we can find the scan dataset we will be processing today?\n"
    "Example:\n"
    "CT_scan_01\n"
    "This folder name is used to name workflow outputs. \n"
    "If it is very long, your workflow output names will also be very long."    
)

print()
scanpath = ask_existing_path(
    "What is the entire path to the folder for this scan dataset?\n"
    "Please include the dataset folder name itself in the path.\n"
    "Example:\n"
    "C:/MyProject/CT_scan_01 \n"
    "You will be asked for this filepath again in subsequent workflow steps, so keep it handy. \n",
    is_dir=True
)


slicepath = ask_existing_path(
    "What is the full path to the folder containing the slice files?\n"
    "Please include the slice folder itself in the path.\n"
    "Example:\n"
    "C:/MyProject/CT_scan_01/Slices",
    is_dir=True
)

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


    print()
    print(f"We found {n_slices} supported slice files.")
    print("First 5 slices:", [os.path.basename(f) for f in slice_files[:5]])
    print("Last 5 slices:", [os.path.basename(f) for f in slice_files[-5:]])
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
        break

    print()
    print("Let's try a different slice folder.")
    print()

    slicepath = ask_existing_path(
        "What is the full path to the folder containing the slice files?",
        is_dir=True
    )

layoutfile = ask_existing_path(
    "What is the path to the layout CSV file?\n"
    "Please include the filename in the path.\n"
    "This file tells the workflow which specimens are expected in the scan and where they are located within the scan. \n"
    "The layout CSV should describe this dataset only and contain: \n"
    "  Column 1: Tier number \n"
    "  Column 2: Row number  \n"
    "  Column 3+: Specimen identifiers\n",
    is_dir=False
)

layout_filename = os.path.basename(layoutfile)

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

output_path = create_output_folder(scanpath, dataset_folder_name)

print()
print(f"All workflow outputs will be saved here:")
print(output_path)
print()

print(
    "Now you need to choose a repesesntative slice from the slice folder for previewing. \n" 
    "The next step in the workflow will ask you to compare that slice to the layout CSV, \n"
    "and then based on what you see, select a few settings."
)

print()
slice_index_fraction = ask_float_in_range(
    "Choose a representative slice for previewing in the next step.\n"
    "This should land near the middle of a tier that contains specimens.\n"
    "For one occupied tier, 0.50 is often good. For two occupied tiers, 0.25 or 0.75 may be better.\n"
    "Avoid choosing a value that falls within an empty tier.\n"
    "If the scan appears upside down relative to the layout later, we will correct that later in the workflow.\n"
    "The default is 0.50. Press Enter to accept the default, or type a different value.",
    minimum=0.0,
    maximum=1.0,
    default=0.50
)


timer_00_stop = timeit.default_timer()

# ============================================================
# Calculate runtimes
# ============================================================

runtime_00_seconds = timer_00_stop - timer_00_start
print("00_share_data.py runtime: ", runtime_00_seconds)

# ============================================================
# Create metadata file and dictionaries
# ============================================================

metadata_filename = build_metadata_filename(dataset_folder_name)
metadata_path = os.path.join(output_path, metadata_filename)  

metadata= {}

# ============================================================
# Update metadata
# ============================================================

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

    "runtime_00_seconds": runtime_00_seconds,
}

save_metadata(metadata_path, metadata)


# ============================================================
# Confirm completion
# ============================================================

print()
print("Thank you for sharing the data.")
print()
print("Setup complete.")
print(f"Metadata saved to:")
print(metadata_path)
print()

ask_run_next_step("01_set_rotation_crop.py", scanpath, metadata_path)




