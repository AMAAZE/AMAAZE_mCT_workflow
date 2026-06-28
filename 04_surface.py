#!/usr/bin/env python3

"""
04_surface.py

Original processing logic: RileyWilde
Refactoring and workflow design: Katrina E. Yezzi-Woodley

Use the extraction plan to reconstruct per-specimen subvolumes
from the original slice stack, generate surface meshes, and clean those meshes.

Subvolumes are saved as compressed arrays with voxel scaling metadata.
Meshes are written to the Meshes folder.
Cleaned meshes are written to the Clean_Meshes folder.
"""

# ============================================================
# Configuration and imports
# ============================================================

from utils import *

# ============================================================
# Load metadata path(s)
# ============================================================

metadata_paths = get_metadata_paths_from_command_line_or_user(
    step_name="04_surface",
    allow_batch=True
)

# ============================================================
# Process one workflow run at a time
# ============================================================

for metadata_path in metadata_paths:

    timer_04_start = timeit.default_timer()

    metadata = load_metadata_if_available(metadata_path)
    
    md = unpack_metadata(metadata)

    print()
    print("Surfacing and cleaning workflow for:")
    print(metadata_path)
    print()

    # ============================================================
    # Load subvolume, extraction plan, and slices
    # ============================================================

    try:
        npzdata = np.load(md.subvolume_file)

    except FileNotFoundError:
        raise RuntimeError(
            f"Processed volume not found:\n{md.subvolume_file}\n\n"
            "Run 02_build_subvolume.py to create the processed volume."
        )

    try:
        extraction_plan = pd.read_csv(md.extraction_plan_csv)

    except FileNotFoundError:
        raise RuntimeError(
            f"Extraction plan not found:\n{md.extraction_plan_csv}\n\n"
            "Run 03_segment.py to create the extraction plan."
        )

    slice_files, slice_indices = get_sorted_slice_files(md.slicepath)
    
    #Can we add something to the above exceptions that make sure it goes on and tries the next json?

    # ============================================================
    # Prepare output folders
    # ============================================================

    outpath = os.path.join(md.output_path, "Meshes")
    os.makedirs(outpath, exist_ok=True)

    clean_mesh_folder = os.path.join(md.output_path, "Clean_Meshes")
    os.makedirs(clean_mesh_folder, exist_ok=True)

    mesh_cleaning_log_csv = os.path.join(
        md.output_path,
        "mesh_cleaning_log.csv"
    )
    
    surfacing_errors_csv = os.path.join(md.output_path, "surfacing_errors.csv")

    # ============================================================
    # Group extraction plan by tier
    # ============================================================

    tier_groups = extraction_plan.groupby("tier_id")

    # ============================================================
    # Set voxel geometry
    # ============================================================

    if md.is_isotropic:
        voxel_spacing_mm = md.voxel_size_mm
    else:
        voxel_spacing_mm = md.voxel_spacing_mm

# ============================================================
# Set parallelization
# ============================================================

    print()
    print(
        "Surfacing can use multiple CPU cores (parallelization) "
        "to speed up extraction, surfacing, and mesh cleaning."
    )
    print("Most users should accept the defaults.")
    print("Advanced users may customize CPU use for each process.")
    print()

    extract_parallelization = choose_parallel_cores(
        process_label="specimen extraction",
        default_percentage=85,
    )

    surface_parallelization = choose_parallel_cores(
        process_label="surfacing",
        default_percentage=95,
    )

    clean_parallelization = choose_parallel_cores(
        process_label="mesh cleaning",
        default_percentage=25,
    )

    extract_num_cores = extract_parallelization["num_cores"]
    surface_num_cores = surface_parallelization["num_cores"]
    clean_num_cores = clean_parallelization["num_cores"]

    print()
    print("Parallelization settings:")
    print(f"Extraction:    {extract_num_cores} core(s)")
    print(f"Surfacing:     {surface_num_cores} core(s)")
    print(f"Mesh cleaning: {clean_num_cores} core(s)")
    print()
    
    print()
    print("Setup is complete.")
    print("AMAAZE will now extract specimen volumes, surface meshes, and clean meshes.")
    print("This can take a while, especially for large scans or many specimens.")
    print("You do not need to answer more questions during this part.")
    print("Progress messages will appear in the terminal as files are completed.")
    print()
    
    # We could probably collapse the two sections into 1 if.
    # CPU settings should be the same across the entire run instance for all datasets so perhaps it shouldn't be in the major loop?

    interactive_setup_timer_stop = timeit.default_timer()
    # We needd to update the timer situation.

    # ============================================================
    # Extract specimen subvolumes from original slice stack
    # ============================================================

    extraction_timer_start = timeit.default_timer()

    for tier_id, tier_plan in tier_groups:

        tier_plan = tier_plan.copy()

        tier_z_range_reduced = [
            int(tier_plan["z_start"].iloc[0]),
            int(tier_plan["z_end"].iloc[0])
        ]

        tier_z_range_original = [
            tier_z_range_reduced[0] * md.zwindow,
            min(tier_z_range_reduced[1] * md.zwindow, len(slice_files))
        ]

        tier_rotation_angle = float(tier_plan["tier_rotation_angle"].iloc[0])

        for slice_index in range(
            tier_z_range_original[0],
            tier_z_range_original[1]
        ):

            image = read_slice(slice_files[slice_index])
            image = apply_preview_orientation(image, md.transpose_preview)
            image = rotate(image, md.rotation_angle, preserve_range=True, resize=True)
            image = image[md.rowrng[0]:md.rowrng[1], md.colrng[0]:md.colrng[1]].copy()
            image = rotate(image, tier_rotation_angle, preserve_range=True)

            row_scale = image.shape[0] / npzdata["vol"].shape[1]
            col_scale = image.shape[1] / npzdata["vol"].shape[2]

            if slice_index == tier_z_range_original[0]:

                tier_plan["row_start_scaled"] = (
                    tier_plan["row_start"].astype(float) * row_scale
                ).round().astype(int)

                tier_plan["row_end_scaled"] = (
                    tier_plan["row_end"].astype(float) * row_scale
                ).round().astype(int)

                tier_plan["col_start_scaled"] = (
                    tier_plan["col_start"].astype(float) * col_scale
                ).round().astype(int)

                tier_plan["col_end_scaled"] = (
                    tier_plan["col_end"].astype(float) * col_scale
                ).round().astype(int)

                tier_plan["row_start_padded"] = np.maximum(
                    tier_plan["row_start_scaled"] - md.padding,
                    0
                )

                tier_plan["row_end_padded"] = np.minimum(
                    tier_plan["row_end_scaled"] + md.padding,
                    image.shape[0]
                )

                tier_plan["col_start_padded"] = np.maximum(
                    tier_plan["col_start_scaled"] - md.padding,
                    0
                )

                tier_plan["col_end_padded"] = np.minimum(
                    tier_plan["col_end_scaled"] + md.padding,
                    image.shape[1]
                )

            Parallel(n_jobs=extract_num_cores)(
                delayed(extract_specimen_subvolume_slice)(
                    slice_index=slice_index,
                    specimen_row=specimen_row,
                    tier_z_start=tier_z_range_original[0],
                    image=image,
                    outpath=outpath
                )
                for _, specimen_row in tier_plan.iterrows()
            )

    # ============================================================
    # Save specimen overview images and compressed subvolume files
    # ============================================================

    for specimen_id in extraction_plan["specimen_id"]:

        specimen_id = str(specimen_id)
        fname = os.path.join(outpath, specimen_id)

        IMAGES = np.load(fname + ".npy", mmap_mode="r")

        print(
            specimen_id,
            "intensity range:",
            IMAGES.min(),
            IMAGES.max()
        )

        overview = dicom.bone_overview(IMAGES)
        plt.imsave(fname + ".png", overview, cmap="gray")

        shape_out = IMAGES.shape
        np.savez_compressed(fname, I=IMAGES, voxel_size_mm=md.voxel_size_mm, voxel_spacing_mm=voxel_spacing_mm)

        del IMAGES
        os.remove(fname + ".npy")

        print("finished ", fname, " size: ", shape_out)

    extraction_timer_stop = timeit.default_timer()

    # ============================================================
    # Surface specimen volumes
    # ============================================================

    surfacing_timer_start = timeit.default_timer()

    print()
    print("Starting specimen surfacing.")
    print("This step creates mesh files from the extracted specimen volumes.")
    print()

    dicom.surface_bones_parallel(
        directory=outpath,
        iso=md.iso,
        error_fname=surfacing_errors_csv,
        ncores=surface_num_cores
    )

    n_surfacing_errors = count_csv_rows(surfacing_errors_csv)

    n_specimens_extracted = len(extraction_plan["specimen_id"].unique())

    n_meshes_generated = len([
        f for f in os.listdir(outpath)
        if f.lower().endswith(".ply")
    ])

    surfacing_timer_stop = timeit.default_timer()

    # ============================================================
    # Clean generated meshes
    # ============================================================

    cleaning_timer_start = timeit.default_timer()

    print()
    print("Starting mesh cleaning.")
    print("This step removes small disconnected fragments from generated meshes.")
    print()

    input_mesh_folder = outpath

    mesh_files = os.listdir(input_mesh_folder)

    mesh_filenames = [
        f for f in mesh_files
        if f.lower().endswith(".ply")
    ]

    if len(mesh_filenames) == 0:
        raise RuntimeError(
            f"No .ply files found in {input_mesh_folder}. "
            "Surfacing may have failed; check surfacing errors."
        )

    cleaning_results = Parallel(n_jobs=clean_num_cores)(
        delayed(clean_mesh_file)(
            f,
            input_mesh_folder,
            clean_mesh_folder,
            md.dust_cutoff,
            md.hole_tolerance
        )
        for f in mesh_filenames
    )

    mesh_cleaning_log = pd.DataFrame(cleaning_results)

    mesh_cleaning_log.to_csv(
        mesh_cleaning_log_csv,
        index=False
    )

    n_meshes_cleaned = int(
        (mesh_cleaning_log["status"] == "success").sum()
    )

    n_mesh_cleaning_failures = int(
        (mesh_cleaning_log["status"] == "failure").sum()
    )

    cleaning_timer_stop = timeit.default_timer()
    timer_04_stop = timeit.default_timer()

    # ============================================================
    # Calculate runtimes
    # ============================================================

    runtime_04_seconds = (timer_04_stop - timer_04_start)
    extraction_runtime_seconds = (extraction_timer_stop - extraction_timer_start)
    surfacing_runtime_seconds = (surfacing_timer_stop - surfacing_timer_start)
    mesh_cleaning_runtime_seconds = (cleaning_timer_stop - cleaning_timer_start)
    
    total_workflow_runtime_seconds = (
        md.runtime_00_seconds +
        md.runtime_01_seconds +
        md.runtime_02_seconds +
        md.runtime_03_seconds +
        runtime_04_seconds
    )
    
    automated_runtime_seconds = (
        md.runtime_02_seconds + 
        runtime_04_seconds
    )
    
    interactive_runtime_seconds = (
        total_workflow_runtime_seconds - 
        automated_runtime_seconds
    )      

    # ============================================================
    # Update metadata
    # ============================================================

    metadata.setdefault("workflow_runtimes", {})
    metadata["workflow_runtimes"]["runtime_04_seconds"] = runtime_04_seconds
    metadata["workflow_runtimes"]["extraction_runtime_seconds"] = extraction_runtime_seconds
    metadata["workflow_runtimes"]["surfacing_runtime_seconds"] = surfacing_runtime_seconds
    metadata["workflow_runtimes"]["mesh_cleaning_runtime_seconds"] = mesh_cleaning_runtime_seconds
    metadata["workflow_runtimes"]["total_workflow_runtime_seconds"] = total_workflow_runtime_seconds 
    metadata["workflow_runtimes"]["automated_runtime_seconds"] = automated_runtime_seconds
    metadata["workflow_runtimes"]["interactive_runtime_seconds"] = interactive_runtime_seconds
    
    metadata["04_surface"] = {
        "status": "complete",

        "mesh_folder": outpath,
        "clean_mesh_folder": clean_mesh_folder,
        "surfacing_errors_csv": surfacing_errors_csv,
        "mesh_cleaning_log_csv": mesh_cleaning_log_csv,

        "n_specimens_extracted": n_specimens_extracted,
        "n_meshes_generated": n_meshes_generated,
        "n_surfacing_errors": n_surfacing_errors,
        "n_input_meshes": len(mesh_filenames),
        "n_meshes_cleaned": n_meshes_cleaned,
        "n_mesh_cleaning_failures": n_mesh_cleaning_failures,

        "parallelization": {
            "extraction": extract_parallelization,
            "surfacing": surface_parallelization,
            "mesh_cleaning": clean_parallelization,
        },
    }

    save_metadata(metadata_path, metadata)

    # ============================================================
    # Confirm completion for this workflow
    # ============================================================

    print()
    print("Surfacing and cleaning complete.")
    print(f"Surface meshes were saved to:\n{outpath}")
    print(f"Cleaned meshes were saved to:\n{clean_mesh_folder}")
    print("Metadata updated:")
    print(metadata_path)
    print()

# ============================================================
# Confirm completion for all workflows
# ============================================================

if len(metadata_paths) > 1:
    print()
    print("Batch surfacing and cleaning complete.")
    print(f"Processed {len(metadata_paths)} workflow(s).")
    print()

print()
print("You've reached the end of the primary workflow. Well done!")
print()
