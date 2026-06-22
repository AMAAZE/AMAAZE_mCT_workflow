#!/usr/bin/env python3

"""
04_surface.py

Original processing logic: RileyWilde
Refactoring and workflow design: Katrina E. Yezzi-Woodley
Use the extraction plan to reconstruct per-specimen subvolumes
from the original slice stack, then generate surface meshes for each specimen.

Subvolumes are saved as compressed arrays with voxel scaling metadata,
and meshes are generated via marching cubes and written to the Meshes folder.
"""

# ============================================================
# Configuration and imports
# ============================================================

from utils import *

# ============================================================
# Load metadata
# ============================================================
timer_04_start = timeit.default_timer()

print()
dataset_path = ask_existing_path(
    "What is the full path to the dataset folder you want to continue working on?\n"
    "This should be the same dataset folder path you gave to 00_share_data.py.\n"
    "Example:\n"
    "C:/MyProject/CT_scan_01",
    is_dir=True
)

metadata_path = find_metadata_file_in_dataset(dataset_path)
metadata = load_metadata_if_available(metadata_path)

slicepath = metadata["00_share_data"]["slicepath"]
output_path = metadata["00_share_data"]["output_path"]

voxel_size_mm = metadata["00_share_data"]["voxel_size_mm"]
voxel_spacing_mm = metadata["00_share_data"]["voxel_spacing_mm"]

rotation_angle = metadata["01_set_rotation_crop"]["rotation_angle"]
transpose_preview = metadata["01_set_rotation_crop"]["transpose_preview"]
rowrng = metadata["01_set_rotation_crop"]["rowrng"]
colrng = metadata["01_set_rotation_crop"]["colrng"]

subvolume_file = metadata["02_build_subvolume"]["subvolume_file"]

zwindow = metadata["02_build_subvolume"]["zwindow"]

extraction_plan_csv = metadata["03_segment"]["extraction_plan_csv"]

iso = metadata["03_segment"]["surfacing_setup"]["iso"]
padding = metadata["03_segment"]["surfacing_setup"]["padding"]


# ============================================================
# Load subvolume, extraction plan, and slices
# ============================================================

try:
    npzdata = np.load(subvolume_file)

except FileNotFoundError:
    raise RuntimeError(
        f"Processed volume not found:\n{subvolume_file}\n\n"
        "Run 02_build_subvolume.py to create the processed volume."
    )

try:
    extraction_plan = pd.read_csv(extraction_plan_csv)

except FileNotFoundError:
    raise RuntimeError(
        f"Extraction plan not found:\n{extraction_plan_csv}\n\n"
        "Run 03_segment.py to create the extraction plan."
    )

slice_files, slice_indices = get_sorted_slice_files(slicepath)


# ============================================================
# Prepare output folders
# ============================================================

outpath = os.path.join(output_path, "Meshes")
os.makedirs(outpath, exist_ok=True)

# ============================================================
# Group extraction plan by tier
# ============================================================

tier_groups = extraction_plan.groupby("tier_id")

# ============================================================
# Set voxel geometry
# ============================================================

dx = voxel_size_mm

if voxel_spacing_mm is None:
    dz = dx
else:
    dz = voxel_spacing_mm


# ============================================================
# Set parallelization
# ============================================================

default_extract_ncores = max(1, int(multiprocessing.cpu_count() * 0.85))
default_surface_ncores = min(20, multiprocessing.cpu_count())

print()
print("This step can use multiple CPU cores (parallelization) to speed up extraction and surfacing.")
print(
    f"By default, extraction uses approximately 85% of available CPU cores "
    f"({default_extract_ncores} cores on this computer), and surfacing uses "
    f"up to 100% of available CPU cores with a maximum of 20 "
    f"({default_surface_ncores} cores on this computer)."
)
print("Advanced users may choose custom values if desired.")
print()

custom_cores = ask_yes_no(
    "Do you want to manually set CPU core counts for this step?\n"
    "Most users should choose no.",
    default="n"
)

if custom_cores:
    extract_num_cores = ask(
        "How many cores for subvolume extraction?",
        default=default_extract_ncores,
        cast=int
    )
    surface_num_cores = ask(
        "How many cores for surfacing?",
        default=default_surface_ncores,
        cast=int
    )
else:
    extract_num_cores = default_extract_ncores
    surface_num_cores = default_surface_ncores
    
extract_num_cores_method = "manual" if custom_cores else "default"
surface_num_cores_method = "manual" if custom_cores else "default"
    
print()
print("Setup is complete. AMAAZE will now extract specimen volumes and surface them.")
print("This can take a while, especially for large scans or many specimens.")
print("You do not need to answer more questions during this part.")
print("Progress messages will appear in the terminal as files are completed.")
print()

interactive_setup_timer_stop = timeit.default_timer()

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
        tier_z_range_reduced[0] * zwindow,
        min(tier_z_range_reduced[1] * zwindow, len(slice_files))
    ]

    tier_rotation_angle = float(tier_plan["tier_rotation_angle"].iloc[0])
    
    for slice_index in range(
        tier_z_range_original[0],
        tier_z_range_original[1]
    ):

        image = read_slice(slice_files[slice_index])
        image = apply_preview_orientation(image, transpose_preview)
        image = rotate(image, rotation_angle, preserve_range=True, resize=True)
        image = image[rowrng[0]:rowrng[1], colrng[0]:colrng[1]].copy()
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
                tier_plan["row_start_scaled"] - padding,
                0
            )

            tier_plan["row_end_padded"] = np.minimum(
                tier_plan["row_end_scaled"] + padding,
                image.shape[0]
            )

            tier_plan["col_start_padded"] = np.maximum(
                tier_plan["col_start_scaled"] - padding,
                0
            )

            tier_plan["col_end_padded"] = np.minimum(
                tier_plan["col_end_scaled"] + padding,
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

    print("HELLO I AM INSIDE THE IMAGES LOOP")
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
    np.savez_compressed(fname, I=IMAGES, dx=dx, dz=dz)

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

surfacing_errors_csv = os.path.join(output_path, "surfacing_errors.csv")

dicom.surface_bones_parallel(
    directory=outpath,
    iso=iso,
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
timer_04_stop = timeit.default_timer()

# ============================================================
# Calculate runtimes
# ============================================================

interactive_setup_runtime_seconds = (
    interactive_setup_timer_stop - timer_04_start
)

extraction_runtime_seconds = (
    extraction_timer_stop - extraction_timer_start
)

surfacing_runtime_seconds = (
    surfacing_timer_stop - surfacing_timer_start
)

total_runtime_04_seconds = (
    timer_04_stop - timer_04_start
)

automated_runtime_04_seconds = (
    extraction_runtime_seconds
    + surfacing_runtime_seconds
)

# ============================================================
# Update metadata
# ============================================================

metadata["04_surface"] = {
    "status": "complete",

    "mesh_folder": outpath,
    "surfacing_errors_csv": surfacing_errors_csv,

    "n_specimens_extracted": n_specimens_extracted,
    "n_meshes_generated": n_meshes_generated,
    "n_surfacing_errors": n_surfacing_errors,

    "parallelization": {
        "extract_num_cores": extract_num_cores,
        "extract_num_cores_method": extract_num_cores_method,
        "surface_num_cores": surface_num_cores,
        "surface_num_cores_method": surface_num_cores_method,
    },

    "runtime_seconds": {
        "interactive_setup": interactive_setup_runtime_seconds,
        "extraction": extraction_runtime_seconds,
        "surfacing": surfacing_runtime_seconds,
        "total_runtime_seconds": total_runtime_04_seconds,
        "automated_runtime_seconds": automated_runtime_04_seconds,
    },
}

save_metadata(metadata_path, metadata)

# ============================================================
# Confirm completion
# ============================================================

print()
print("Surfacing complete.")
print(f"Surface meshes were saved to:\n{outpath}")
print()
print("Before continuing, open and inspect a few meshes.")
print("Verify that specimens are complete, properly scaled, and free of obvious extraction or surfacing problems.")
print()
print("If the meshes look reasonable, continue to:")
print("python 05_clean_meshes.py")
print()


