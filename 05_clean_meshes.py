#!/usr/bin/env python3

"""
05_clean_meshes.py

Original processing logic: RileyWilde
Refactoring and workflow design: Katrina E. Yezzi-Woodley

Post-process generated meshes by removing small disconnected components
and retaining the primary specimen geometry.

Cleaned meshes are written to the Clean_Meshes folder, ensuring more stable
and interpretable geometry for downstream analysis.
"""

# ============================================================
# Configuration and imports
# ============================================================

from utils import *

# ============================================================
# Load metadata
# ============================================================

timer_05_start = timeit.default_timer()

dataset_path = ask_existing_path(
    "What is the name of the dataset folder you want to continue working on?\n"
    "This should be the same dataset folder you gave to 00_share_data.py.",
    is_dir=True
)

metadata_path = find_metadata_file_in_dataset(dataset_path)
metadata = load_metadata_if_available(metadata_path)

output_path = metadata["00_share_data"]["output_path"]

input_mesh_folder = metadata["04_surface"]["mesh_folder"]

clean_mesh_folder = os.path.join(output_path, "Clean_Meshes")
os.makedirs(clean_mesh_folder, exist_ok=True)

mesh_cleaning_log_csv = os.path.join(
    output_path,
    "mesh_cleaning_log.csv"
)

# ============================================================
# Set mesh-cleaning parameters
# ============================================================

print()
print("This step removes small disconnected mesh fragments from each surface mesh")
print("and retains the primary specimen geometry.")
print()
print("Cleaning decisions are recorded in a mesh-cleaning log so that each")
print("mesh can be reviewed later if the cleaned output looks unexpected.")
print()

default_clean_ncores = max(
    1,
    int(multiprocessing.cpu_count() / 4)
)

custom_cores = ask_yes_no(
    "By default, mesh cleaning uses approximately one quarter of the \n"
    "available CPU cores, but never fewer than one core. \n"
    f"On this computer, the default is {default_clean_ncores} cores. \n" 
    "Most users choose the default. \n"
    "Do you want to manually set CPU core count for this step?\n",
    default="n"
)

if custom_cores:
    num_cores = ask(
        "How many CPU cores would you like to use for mesh cleaning?",
        default=default_clean_ncores,
        cast=int
    )
else:
    num_cores = default_clean_ncores
    
clean_num_cores_method = "manual" if custom_cores else "default"
    
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
        f"No .ply files found in {input_mesh_folder}. "
        "Run 04_surface.py first or check surfacing errors."
    )

# ============================================================
# Ask for cleaning parameters
# ============================================================

dust_cutoff = ask(
    "Dust cutoff controls how small disconnected mesh fragments are removed.\n"
    "The default is 20 vertices. Most users should press Enter.",
    default=20,
    cast=int
)

hole_tolerance = ask(
    "Hole tolerance controls how many detected holes are allowed when selecting the main mesh component.\n"
    "The default is 0. Most users should press Enter.",
    default=0,
    cast=int
)

interactive_setup_timer_stop = timeit.default_timer()

# ============================================================
# Run cleaning
# ============================================================

cleanup_timer_start = timeit.default_timer()

cleaning_results = Parallel(n_jobs=num_cores)(
    delayed(clean_mesh_file)(
        f,
        input_mesh_folder,
        clean_mesh_folder,
        dust_cutoff,
        hole_tolerance
    )
    for f in mesh_filenames
)

# ==========================================================
# Write mesh_cleaning_log.csv
# ===========================================================

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

cleanup_timer_stop = timeit.default_timer()
timer_05_stop = timeit.default_timer()

# ============================================================
# Calculate runtimes
# ============================================================

interactive_setup_runtime_seconds = (
    interactive_setup_timer_stop - timer_05_start
)

automated_mesh_cleaning_runtime_seconds = (
    cleanup_timer_stop - cleanup_timer_start
)

runtime_05_seconds = (
    timer_05_stop - timer_05_start
)

# ============================================================
# Update metadata
# ============================================================

metadata["05_clean_meshes"] = {
    "status": "complete",

    "input_mesh_folder": input_mesh_folder,
    "clean_mesh_folder": clean_mesh_folder,
    "mesh_cleaning_log_csv": mesh_cleaning_log_csv,

    "mesh_cleaning_parameters": {
        "dust_cutoff": dust_cutoff,
        "hole_tolerance": hole_tolerance,
    },

    "parallelization": {
        "num_cores": num_cores,
        "clean_num_cores_method": clean_num_cores_method,
    },

    "n_input_meshes": len(mesh_filenames),
    "n_meshes_cleaned": n_meshes_cleaned,
    "n_mesh_cleaning_failures": n_mesh_cleaning_failures,

    "runtime_seconds": {
        "interactive_setup": interactive_setup_runtime_seconds,
        "automated_mesh_cleaning": automated_mesh_cleaning_runtime_seconds,
        "total_runtime_seconds": runtime_05_seconds,
    },
}
save_metadata(metadata_path, metadata)

# ============================================================
# Final run report
# ============================================================

report_outputs = write_final_run_report(metadata)

print()
print("Final rendered Markdown report written to:")
print(report_outputs["run_report_markdown"])

if report_outputs["pdf_created"]:
    print()
    print("Final PDF report written to:")
    print(report_outputs["run_report_pdf"])
else:
    print()
    print("WARNING: Pandoc was not found, so the final PDF report was not created.")
    print("The rendered Markdown report was still created successfully.")

metadata["05_clean_meshes"]["run_report_template_md"] = report_outputs["run_report_template_md"]
metadata["05_clean_meshes"]["run_report_markdown"] = report_outputs["run_report_markdown"]
metadata["05_clean_meshes"]["run_report_pdf"] = report_outputs["run_report_pdf"]
metadata["05_clean_meshes"]["report_timestamp"] = report_outputs["report_timestamp"]
metadata["05_clean_meshes"]["report_timestamp_human"] = report_outputs["report_timestamp_human"]


save_metadata(metadata_path, metadata)

# ============================================================
# Confirm completion
# ============================================================

print()
print("Mesh cleaning complete.")
print(f"Cleaned meshes were saved to:\n{clean_mesh_folder}")
print()

print()
print("Final rendered Markdown report:")
print(report_outputs["run_report_markdown"])

if report_outputs["pdf_created"]:
    print()
    print("Final PDF report:")
    print(report_outputs["run_report_pdf"])

print()
print("The final run report is the primary easy-to-read summary of this workflow run.")
print("The metadata JSON remains the primary machine-readable reproducibility record.")
print()

print("IMPORTANT:")
print(
    "If you plan to run this workflow again and wish to preserve all outputs "
    "from the current run, rename or archive the current output folder before "
    "starting a new run."
)
print(
    "Subsequent runs may overwrite workflow outputs, metadata files, and reports."
)
print()

print("Metadata updated:")
print(metadata_path)
print()

print("Next (optional) step:")
print("python 06_render_views.py, \n"
    "python 07_build_contact_sheet.py \n"
    "python 08_render_gifs.py, and/or \n"
    "python 09_reproduce_run.py, \n"
    "NOTE: Rename or archive last output directory to rerun flow."
)
