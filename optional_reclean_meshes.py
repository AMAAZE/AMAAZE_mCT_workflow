#!/usr/bin/env python3

"""
reclean_meshes.py

Optional accessory utility for the AMAAZE mCT workflow.

This script re-runs mesh cleaning on meshes produced by the canonical
04_surface.py workflow without modifying the canonical workflow metadata JSON.

Each re-cleaning attempt is written to its own timestamped output folder,
along with a cleaning log CSV and a small sidecar metadata JSON describing
the re-cleaning run.
"""

# ============================================================
# Configuration and imports
# ============================================================

from utils import *

# ============================================================
# Load canonical workflow metadata
# ============================================================

timer_reclean_start = timeit.default_timer()

print()
print("AMAAZE optional mesh re-cleaning utility")
print()
print("This utility re-cleans meshes that were already produced by the main workflow.")
print("It reads the workflow metadata file for provenance, but it does not modify it.")
print()

metadata_path = ask_existing_path(
    "What is the full path to the workflow metadata JSON file?\n"
    "This is the metadata file created by the main AMAAZE workflow.",
    is_dir=False
)

metadata = load_metadata_if_available(metadata_path)

if metadata is None:
    raise RuntimeError(
        f"Could not load metadata file:\n{metadata_path}"
    )

output_path = metadata["00_share_data"]["output_path"]
dataset_folder_name = metadata["00_share_data"]["dataset_folder_name"]

input_mesh_folder = metadata["04_surface"]["mesh_folder"]

if not os.path.isdir(input_mesh_folder):
    raise RuntimeError(
        f"Input mesh folder was not found:\n{input_mesh_folder}\n\n"
        "Check that 04_surface.py completed successfully."
    )

# ============================================================
# Review previous cleaning parameters
# ============================================================

previous_cleaning_parameters = {}

if "mesh_cleaning_parameters" in metadata.get("04_surface", {}):
    previous_cleaning_parameters = metadata["04_surface"]["mesh_cleaning_parameters"]

elif "mesh_cleaning_parameters" in metadata.get("03_segment", {}):
    previous_cleaning_parameters = metadata["03_segment"]["mesh_cleaning_parameters"]

previous_dust_cutoff = previous_cleaning_parameters.get("dust_cutoff", 20)
previous_hole_tolerance = previous_cleaning_parameters.get("hole_tolerance", 0)

print()
print("Previous cleaning parameters found in the workflow metadata:")
print(f"    dust_cutoff:    {previous_dust_cutoff}")
print(f"    hole_tolerance: {previous_hole_tolerance}")
print()

reuse_previous_parameters = ask_yes_no(
    "Use these same cleaning parameters for this re-cleaning run?",
    default="y"
)

if reuse_previous_parameters:
    dust_cutoff = previous_dust_cutoff
    hole_tolerance = previous_hole_tolerance
else:
    dust_cutoff = ask(
        "Dust cutoff controls how small disconnected mesh fragments are removed.\n"
        "Enter the dust cutoff for this re-cleaning run.",
        default=previous_dust_cutoff,
        cast=int
    )

    hole_tolerance = ask(
        "Hole tolerance controls how many detected holes are allowed when selecting the main mesh component.\n"
        "Enter the hole tolerance for this re-cleaning run.",
        default=previous_hole_tolerance,
        cast=int
    )

# ============================================================
# Set parallelization
# ============================================================

default_clean_ncores = max(
    1,
    int(multiprocessing.cpu_count() / 4)
)

print()
print("Mesh re-cleaning can use multiple CPU cores.")
print(
    f"By default, this utility uses approximately one quarter of available CPU cores "
    f"({default_clean_ncores} cores on this computer)."
)
print()

custom_cores = ask_yes_no(
    "Do you want to manually set the CPU core count for this re-cleaning run?",
    default="n"
)

if custom_cores:
    clean_num_cores = ask(
        "How many CPU cores would you like to use for mesh re-cleaning?",
        default=default_clean_ncores,
        cast=int
    )
else:
    clean_num_cores = default_clean_ncores

clean_num_cores_method = "manual" if custom_cores else "default"

interactive_setup_timer_stop = timeit.default_timer()

# ============================================================
# Prepare output folder
# ============================================================

timestamp = current_timestamp_for_filename()

reclean_root_folder = os.path.join(
    output_path,
    "Optional_Reclean_Meshes"
)

reclean_run_folder = os.path.join(
    reclean_root_folder,
    f"reclean_{dataset_folder_name}_{timestamp}"
)

reclean_mesh_folder = os.path.join(
    reclean_run_folder,
    "Recleaned_Meshes"
)

os.makedirs(reclean_mesh_folder, exist_ok=True)

reclean_log_csv = os.path.join(
    reclean_run_folder,
    f"reclean_meshes_log_{dataset_folder_name}_{timestamp}.csv"
)

reclean_metadata_json = os.path.join(
    reclean_run_folder,
    f"reclean_meshes_metadata_{dataset_folder_name}_{timestamp}.json"
)

# ============================================================
# Discover input meshes
# ============================================================

mesh_files = os.listdir(input_mesh_folder)

mesh_filenames = [
    f for f in mesh_files
    if f.lower().endswith(".ply")
]

if len(mesh_filenames) == 0:
    raise RuntimeError(
        f"No .ply files found in:\n{input_mesh_folder}\n\n"
        "Run 04_surface.py first or check surfacing errors."
    )

print()
print(f"Found {len(mesh_filenames)} mesh file(s) to re-clean.")
print("Re-cleaned meshes will be saved to:")
print(reclean_mesh_folder)
print()

# ============================================================
# Run cleaning
# ============================================================

cleanup_timer_start = timeit.default_timer()

cleaning_results = Parallel(n_jobs=clean_num_cores)(
    delayed(clean_mesh_file)(
        f,
        input_mesh_folder,
        reclean_mesh_folder,
        dust_cutoff,
        hole_tolerance
    )
    for f in mesh_filenames
)

cleanup_timer_stop = timeit.default_timer()
timer_reclean_stop = timeit.default_timer()

# ============================================================
# Write re-cleaning log
# ============================================================

mesh_cleaning_log = pd.DataFrame(cleaning_results)

mesh_cleaning_log.to_csv(
    reclean_log_csv,
    index=False
)

n_meshes_cleaned = int(
    (mesh_cleaning_log["status"] == "success").sum()
)

n_mesh_cleaning_failures = int(
    (mesh_cleaning_log["status"] == "failure").sum()
)

# ============================================================
# Write sidecar re-cleaning metadata
# ============================================================

interactive_setup_runtime_seconds = (
    interactive_setup_timer_stop - timer_reclean_start
)

automated_recleaning_runtime_seconds = (
    cleanup_timer_stop - cleanup_timer_start
)

total_runtime_seconds = (
    timer_reclean_stop - timer_reclean_start
)

reclean_metadata = {
    "script": "reclean_meshes.py",
    "status": "complete",

    "timestamp": timestamp,

    "source_workflow_metadata_json": metadata_path,
    "dataset_folder_name": dataset_folder_name,

    "source_mesh_folder": input_mesh_folder,
    "reclean_run_folder": reclean_run_folder,
    "reclean_mesh_folder": reclean_mesh_folder,
    "reclean_log_csv": reclean_log_csv,

    "previous_cleaning_parameters": {
        "dust_cutoff": previous_dust_cutoff,
        "hole_tolerance": previous_hole_tolerance,
    },

    "recleaning_parameters": {
        "dust_cutoff": dust_cutoff,
        "hole_tolerance": hole_tolerance,
    },

    "parallelization": {
        "clean_num_cores": clean_num_cores,
        "clean_num_cores_method": clean_num_cores_method,
    },

    "summary": {
        "n_input_meshes": len(mesh_filenames),
        "n_meshes_cleaned": n_meshes_cleaned,
        "n_mesh_cleaning_failures": n_mesh_cleaning_failures,
    },

    "runtime_seconds": {
        "interactive_setup": interactive_setup_runtime_seconds,
        "automated_recleaning": automated_recleaning_runtime_seconds,
        "total_runtime_seconds": total_runtime_seconds,
    },
}

save_metadata(reclean_metadata_json, reclean_metadata)

# ============================================================
# Confirm completion
# ============================================================

print()
print("Mesh re-cleaning complete.")
print()
print(f"Re-cleaned meshes saved to:\n{reclean_mesh_folder}")
print()
print(f"Re-cleaning log written to:\n{reclean_log_csv}")
print()
print(f"Re-cleaning metadata written to:\n{reclean_metadata_json}")
print()
print("The original workflow metadata JSON was not modified.")
print()
