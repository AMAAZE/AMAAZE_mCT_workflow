#!/usr/bin/env python3

"""
04_surface.py

Original processing logic: RileyWilde
Refactoring and workflow design: Katrina E. Yezzi-Woodley

Mesh reconstruction and cleaning step for the AMAAZE mCT surfacing workflow.

This script reconstructs each specimen from the original-resolution slice stack
using the extraction plan created by 03_segment.py. It then generates surface
meshes, performs automated mesh cleaning, and records workflow outputs.

This step:
- reconstructs per-specimen image volumes,
- generates surface meshes,
- performs automated mesh cleaning,
- records surfacing and cleaning results,
- and updates the workflow metadata JSON.
"""

# ============================================================
# Configuration and imports
# ============================================================

from utils import *

# ============================================================
# Load metadata path(s)
# ============================================================

TOTAL_QUESTIONS_04 = 3

print_terminal_header("Step 5 of 5: Mesh Surfacing and Cleaning")

print("This step surfaces the data, creating per-specimen meshes,")
print("and then cleans each mesh.")
print()
print("First, you will choose whether to process one workflow")
print("or multiple workflows in batch mode.")
print()
print("Then, you will provide the metadata JSON file")
print("for each workflow you choose to process.")
print()
print("Before surfacing begins, you will select CPU settings")
print("that will be used during surfacing and cleaning.")
print()
print("When you're ready, press Enter to begin.")
input("> ")

print_question_header("Workflow Selection")

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

    surface_run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    
    outpath = os.path.join(md.output_path, f"Meshes_{surface_run_timestamp}")
    os.makedirs(outpath, exist_ok=True)

    clean_mesh_folder = os.path.join(md.output_path, f"Clean_Meshes_{surface_run_timestamp}")
    os.makedirs(clean_mesh_folder, exist_ok=True)

    mesh_cleaning_log_csv = os.path.join(
        md.output_path,
        "mesh_cleaning_log.csv"
    )
    
    surfacing_errors_csv = os.path.join(md.output_path, "surfacing_errors.csv")
    
    specimen_intensity_ranges_csv = os.path.join(
        md.output_path,
        "specimen_intensity_ranges.csv"
    )

    specimen_intensity_rows = []
    
    cpu_diagnostics_csv = os.path.join(
        md.output_path,
        f"cpu_diagnostics_{surface_run_timestamp}.csv"
    )

    cpu_sample_interval_seconds = choose_cpu_sample_interval(
        md.total_slice_gb
    )

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
    
    print_question_header("Selecting CPU Settings")
    
    print()
    print("Surfacing can use multiple CPU cores (parallelization)")
    print("to speed up extraction, surfacing, and mesh cleaning.")
    print()
    print("Default CPU settings are based on the logical CPU count reported by your operating system.")
    print()
    print("You may accept the suggested settings or customize")
    print("CPU use for each process.")
    print()

    print_question_header("Extraction Cores", 1, TOTAL_QUESTIONS_04)

    extract_parallelization = choose_parallel_cores(
        process_label="specimen extraction",
        default_percentage=85,
    )

    print_question_header("Surfacing Cores", 2, TOTAL_QUESTIONS_04)
    
    surface_parallelization = choose_parallel_cores(
        process_label="surfacing",
        default_percentage=95,
    )

    print_question_header("Cleaning Cores", 3, TOTAL_QUESTIONS_04)
    
    clean_parallelization = choose_parallel_cores(
        process_label="mesh cleaning",
        default_percentage=25,
    )

    extract_requested_ncores = extract_parallelization["num_cores"]
    surface_requested_ncores = surface_parallelization["num_cores"]
    clean_requested_ncores = clean_parallelization["num_cores"]

    print()
    print("Parallelization settings:")
    print(f"Extraction:    {extract_requested_ncores} core(s)")
    print(f"Surfacing:     {surface_requested_ncores} core(s)")
    print(f"Mesh cleaning: {clean_requested_ncores} core(s)")
    print()
    
    print()
    print("Setup is complete.")
    print("AMAAZE will now extract specimen volumes, surface meshes, and clean meshes.")
    print("This can take a while, especially for large scans or many specimens.")
    print("You do not need to answer more questions during this part.")
    print("Progress messages will appear in the terminal as files are completed.")
    print()
    
    interactive_setup_timer_stop = timeit.default_timer()
    
    cpu_sampler = CPUDiagnosticSampler(
        csv_path=cpu_diagnostics_csv,
        requested_cores_by_phase={
            "extract": extract_requested_ncores,
            "surface": surface_requested_ncores,
            "clean": clean_requested_ncores,
        },
        total_specimens=len(extraction_plan["specimen_id"].unique()),
        sample_interval_seconds=cpu_sample_interval_seconds,
    )

    cpu_sampler.set_phase("extract")
    cpu_sampler.start()

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

            Parallel(n_jobs=extract_requested_ncores)(
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
        
        intensity_min = float(IMAGES.min())
        intensity_max = float(IMAGES.max())

        specimen_intensity_rows.append({
            "specimen_id": specimen_id,
            "intensity_min": intensity_min,
            "intensity_max": intensity_max,
        })

        print(specimen_id, "intensity range:", intensity_min, intensity_max)

        overview = dicom.bone_overview(IMAGES)
        plt.imsave(fname + ".png", overview, cmap="gray")

        shape_out = IMAGES.shape
        np.savez_compressed(fname, I=IMAGES, dx=md.voxel_size_mm, dz=voxel_spacing_mm)

        del IMAGES
        os.remove(fname + ".npy")

        print("finished ", fname, " size: ", shape_out)

    specimen_intensity_ranges = pd.DataFrame(specimen_intensity_rows)
    specimen_intensity_ranges.to_csv(
        specimen_intensity_ranges_csv,
        index=False
    )
    
    cpu_sampler.set_completed_specimens(
        len(extraction_plan["specimen_id"].unique())
    )
    
    extraction_timer_stop = timeit.default_timer()

    # ============================================================
    # Surface specimen volumes
    # ============================================================

    surfacing_timer_start = timeit.default_timer()

    print()
    print("Starting specimen surfacing.")
    print("This step creates mesh files from the extracted specimen volumes.")
    print()
    
    cpu_sampler.set_phase("surface")

    dicom.surface_bones_parallel(
        directory=outpath,
        iso=md.iso,
        error_fname=surfacing_errors_csv,
        ncores=surface_requested_ncores
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
    
    cpu_sampler.set_phase("clean")

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

    cleaning_results = Parallel(n_jobs=clean_requested_ncores)(
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
    cpu_sampler.stop()
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
        "surface_run_timestamp": surface_run_timestamp,

        "mesh_folder": outpath,
        "clean_mesh_folder": clean_mesh_folder,
        "surfacing_errors_csv": surfacing_errors_csv,
        "mesh_cleaning_log_csv": mesh_cleaning_log_csv,
        "specimen_intensity_ranges_csv": specimen_intensity_ranges_csv,
        "cpu_diagnostics_csv": cpu_diagnostics_csv,
        "cpu_sample_interval_seconds": cpu_sample_interval_seconds,

        "n_specimens_extracted": n_specimens_extracted,
        "n_meshes_generated": n_meshes_generated,
        "n_surfacing_errors": n_surfacing_errors,
        "n_input_meshes": len(mesh_filenames),
        "n_meshes_cleaned": n_meshes_cleaned,
        "n_mesh_cleaning_failures": n_mesh_cleaning_failures,

        "parallelization": {
            "extract_requested_ncores": extract_requested_ncores,
            "surface_requested_ncores": surface_requested_ncores,
            "clean_requested_ncores": clean_requested_ncores,
        },
    }

    save_metadata(metadata_path, metadata)

    # ============================================================
    # Confirm completion for this workflow
    # ============================================================

    print_step_complete_header("Step 5 Complete")

    print("Surfacing and cleaning is complete.")
    print()
    print(f"Setup took {format_runtime(runtime_04_seconds)} to complete.")

    print()
    print("Important: Use this metadata JSON to continue the workflow later.")
    print()
    print("Surface meshes were saved to:")
    print(f"    {outpath}")
    print()
    print("Cleaned meshes were saved to:")
    print(f"    {clean_mesh_folder}")
    print()
    print("Metadata updated:")
    print()
    print(f"    {metadata_path}")
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
print("You've reached the end of the workflow. Well done!")
print()
