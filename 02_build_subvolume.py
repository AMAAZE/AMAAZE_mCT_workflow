#!/usr/bin/env python3

"""
02_build_subvolume.py

Original processing logic: RileyWilde
Refactoring and workflow design: Katrina E. Yezzi-Woodley

Automated reduced-volume construction step for the AMAAZE mCT
surfacing workflow.

This script loads one or more workflow metadata JSON files, reads the
corresponding CT slice stacks, applies the orientation and crop selected
in the previous step, and constructs a reduced working volume by averaging
slices within the chosen z-window.

Each averaged slice is resized with aspect-ratio preservation using a
maximum edge length of 225 pixels for computational efficiency.

The resulting subvolume is saved as an .npz file, and the workflow metadata
is updated for downstream segmentation and extraction.
"""

# ============================================================
# Configuration and imports
# ============================================================

from utils import *

# ============================================================
# Load metadata
# ============================================================

preview_parser = argparse.ArgumentParser(add_help=False)
preview_parser.add_argument("--metadata_path", default=None)
preview_args, _ = preview_parser.parse_known_args()

started_automatically = preview_args.metadata_path is not None

if not started_automatically:
    print_terminal_header("Step 3 of 5: Build Reduced Working Volume")

    print("This step builds the reduced working volume used for segmentation.")
    print()
    print("First, you will choose whether to process one workflow")
    print("or multiple workflows in batch mode.")
    print()
    print("Then, you will provide the metadata JSON file")
    print("for each workflow you choose to process.")
    print()
    print("When you're ready, press Enter to begin.")
    input("> ")

    print_question_header("Workflow Selection")

metadata_paths = get_metadata_paths_from_command_line_or_user(
    step_name="02_build_subvolume",
    allow_batch=True
)

for metadata_path in metadata_paths:

    timer_02_start = timeit.default_timer()
    
    metadata = load_metadata_if_available(metadata_path)

    print()
    print("Building subvolume for:")
    print(metadata_path)
    print()
    
    md = unpack_metadata(metadata)

# ============================================================
# Load slices
# ============================================================

    slice_files, slice_indices = get_sorted_slice_files(md.slicepath)


# ============================================================
# build subvolume
# ============================================================

    n_slices = len(slice_files)

    subsampled = []
    
    max_edge = 225

    for i, slice_file in enumerate(slice_files):
        im = read_slice(slice_file).astype(int)
        if i%md.zwindow==0: 
            imstack = im
        else:
            imstack = imstack+im

        if i%md.zwindow==(md.zwindow-1):
            m = imstack.min()
            imstack = apply_preview_orientation(imstack, md.transpose_preview)
            imstack = rotate(imstack, md.rotation_angle, preserve_range=True, resize=True, cval=m)
            print(i / n_slices, os.path.basename(slice_file), m)
            imstack = imstack[md.rowrng[0]:md.rowrng[1], md.colrng[0]:md.colrng[1]].copy() / md.zwindow
        
            imstack = resize_preserve_aspect(imstack, max_edge)
        
            subsampled.append(imstack)

    remainder = 0
    
    if i%md.zwindow!=(md.zwindow-1):
        imstack = apply_preview_orientation(imstack, md.transpose_preview)
        imstack = rotate(imstack, md.rotation_angle, preserve_range=True, resize=True, cval=imstack.min())
        imstack = imstack[md.rowrng[0]:md.rowrng[1], md.colrng[0]:md.colrng[1]].copy() / ((i % md.zwindow) + 1)

        imstack = resize_preserve_aspect(imstack, max_edge)
        subsampled.append(imstack)
        remainder = (i%md.zwindow) +1

    subvolume_file = os.path.join(
        md.output_path,
        f"{md.dataset_folder_name}_subvolume.npz"
    )
    
    np.savez(
        subvolume_file, 
        vol=subsampled, 
        rowrng=md.rowrng, 
        colrng=md.colrng, 
        rotation_angle=md.rotation_angle, 
        original_slice_shape=im.shape, 
        remainder=remainder,
        entire_subvolume_shape=np.array(subsampled).shape,
        subvolume_slice_shape=np.array(subsampled[0].shape),
        transpose_preview=md.transpose_preview
    )

    entire_subvolume_shape = list(np.array(subsampled).shape)
    subvolume_slice_shape = list(np.array(subsampled[0]).shape)

    timer_02_stop = timeit.default_timer()

# ============================================================
# Calculate runtimes
# ============================================================

    runtime_02_seconds = timer_02_stop - timer_02_start


# ============================================================
# Update metadata
# ============================================================

    metadata.setdefault("workflow_runtimes", {})
    metadata["workflow_runtimes"]["runtime_02_seconds"] = runtime_02_seconds

    metadata["02_build_subvolume"] = {
        "status": "complete",

        "max_edge": max_edge,
        
        "remainder": remainder,
    
        "subvolume_file": subvolume_file,
        "entire_subvolume_shape": entire_subvolume_shape,
        "subvolume_slice_shape": subvolume_slice_shape,
    }

    save_metadata(metadata_path, metadata)

# ============================================================
# Confirm completion
# ============================================================
    print_step_complete_header("Step 3 Complete")

    print("Reduced working volume created.")
    print()
    print(f"Subvolume building took {format_runtime(runtime_02_seconds)} to complete.")
    print()
    print("Subvolume saved to:")
    print()
    print(f"    {subvolume_file}")
    print()
    print("Metadata updated:")
    print()
    print(f"    {metadata_path}")
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

