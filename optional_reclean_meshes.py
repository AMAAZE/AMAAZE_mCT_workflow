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

TOTAL_QUESTIONS_rc = 4

print_terminal_header("Optional Script: Re-clean meshes")

print("In this optional step, you will re-clean meshes that were")
print("created by 04_surface.py using different cleaning parameters.")
print()
print("This script writes a separate re-cleaning folder and metadata.")
print("It does not modify the canonical workflow metadata JSON.")
print()
print("When you're ready, press Enter to begin.")
input("> ")

print_question_header("Workflow Metadata JSON", 1, TOTAL_QUESTIONS_rc)

metadata_paths = get_metadata_paths_from_command_line_or_user(
    step_name="optional_reclean_meshes",
    allow_batch=False
)

metadata_path = metadata_paths[0]
metadata = load_metadata_if_available(metadata_path)

md = unpack_metadata(metadata)

output_path = md.output_path
dataset_folder_name = md.dataset_folder_name

# ============================================================
# Select mesh folder
# ============================================================

print_question_header("Mesh folder", 2, TOTAL_QUESTIONS_rc)

mesh_folders = [
    os.path.join(md.output_path, folder)
    for folder in os.listdir(md.output_path)
    if (
        os.path.isdir(os.path.join(md.output_path, folder))
        and folder.lower().startswith("meshes")
    )
]

mesh_folders.sort()

if len(mesh_folders) == 0:
    raise RuntimeError(
        f"No mesh folders were found in:\n{md.output_path}\n\n"
        "Run 04_surface.py first."
    )

if len(mesh_folders) == 1:
    input_mesh_folder = os.path.normpath(mesh_folders[0])
else:
    print()
    print("More than one mesh folder was found.")
    print("Please choose the mesh folder to re-clean.")
    print()

    for i, folder in enumerate(mesh_folders, start=1):
        print(f"{i}. {folder}")

    print()

    choice = ask(
        "Enter the number of the mesh folder to use.",
        cast=int
    )

    if choice < 1 or choice > len(mesh_folders):
        raise RuntimeError("That number is not in the list.")

    input_mesh_folder = os.path.normpath(mesh_folders[choice - 1])
    
print()
print(f"Using mesh folder:")
print(f"    {input_mesh_folder}")
print()

# ============================================================
# Review previous cleaning parameters
# ============================================================

print_question_header("Cleaning parameters", 3, TOTAL_QUESTIONS_rc)

previous_dust_cutoff = md.dust_cutoff
previous_hole_tolerance = md.hole_tolerance

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
# Set requested CPU cores
# ============================================================

default_clean_ncores = max(
    1,
    int(multiprocessing.cpu_count() / 4)
)

print_question_header("Requested CPU cores", 4, TOTAL_QUESTIONS_rc)

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

print_success("Mesh re-cleaning complete.")

print_step_complete_header("Optional mesh re-cleaning complete")

print("Re-cleaned meshes:")
print()
print(f"    {reclean_mesh_folder}")
print()

print("Re-cleaning log:")
print()
print(f"    {reclean_log_csv}")
print()

print("Re-cleaning metadata:")
print()
print(f"    {reclean_metadata_json}")
print()

print(f"Meshes cleaned: {n_meshes_cleaned}")
print(f"Cleaning failures: {n_mesh_cleaning_failures}")
print()
