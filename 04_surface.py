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

print()
dataset_path = ask_existing_path(
    "What is the name of the dataset folder you want to continue working on?\n"
    "This should be the same dataset folder you gave to 00_share_data.py.",
    is_dir=True
)

# ============================================================
# Load metadata
# ============================================================

metadata_path = find_metadata_file_in_dataset(dataset_path)
metadata = load_metadata_if_available(metadata_path)

scanpath = metadata["paths"]["scanpath"]
slicepath = metadata["paths"]["slicepath"]
output_path = metadata["paths"]["output_path"]

dataset_name = metadata["dataset_name"]
scan_num = metadata["scan_num"]

npz_fname = metadata["outputs"]["npz_file"]
extraction_plan_csv = metadata["outputs"]["extraction_plan_csv"]

rotation_angle = metadata["orientation"]["rotation_angle"]
transpose_preview = metadata["orientation"]["transpose_preview"]
rowrng = metadata["cropping"]["rowrng"]
colrng = metadata["cropping"]["colrng"]
voxel_size_mm = metadata["user_choices"]["voxel_size_mm"]
voxel_spacing_mm = metadata["user_choices"]["voxel_spacing_mm"]

# TODO(dev): Below is some legacy code. Do we need to keep any of this and if so, does it need updating to align with our current work?

tier_ranges, tier_ids = np.unique(info[:, 1:3].astype(int), axis=0, return_inverse=True)

rowrng1 = npzdata["rowrng"]
colrng1 = npzdata["colrng"]
ang2rot = npzdata["ang"]
origsz = npzdata["origsz"]
rem = npzdata["remainder"]
transpose_preview = bool(npzdata["transpose_preview"])

# ============================================================
# Load subvolume, extraction plan, and slices
# ============================================================

npzdata = np.load(npz_fname)
extraction_plan = pd.read_csv(extraction_plan_csv)
slice_files, slice_indices = get_sorted_slice_files(slicepath)

if not os.path.exists(npz_fname):
    raise RuntimeError(f"Processed volume not found: {npz_fname}. Run 02_build_subvolume.py first.")

if not os.path.exists(extraction_plan_csv):
    raise RuntimeError(f"Extraction plan not found: {extraction_plan_csv}. Run 03_segment.py first.")

outpath = os.path.join(output_path, "Meshes")
os.makedirs(outpath, exist_ok=True)


# ============================================================
# Set voxel geometry
# ============================================================

dx = voxel_size_mm

if voxel_spacing_mm is None:
    dz = dx
else:
    dz = voxel_spacing_mm
    
# TODO(dev): Below is old code, do we want to retain the print statements? Or just keep this in the background or say, we are beginning the surfacing protocol. You gave use these voxel dimensions before are they correct?

if voxel_spacing_mm is None:
    dz = dx
    print(f"No voxel_spacing_mm provided; using isotropic spacing dz = dx = {dx} mm")
else:
    dz = voxel_spacing_mm
    print(f"Using voxel_spacing_mm from user_inputs.json: dz = {dz} mm")

# ============================================================
# Set ISO value
# ============================================================

print()
print("An isovalue (ISO) is required to surface the scans.")
print(
    "An isovalue (ISO) is the grayscale threshold used to "
    "decide which voxels belong to the specimen and which "
    "belong to the surrounding background."
)
print()
print(
    "Voxels brighter than the threshold are treated as material "
    "and voxels darker than the threshold are treated as background "
)
print()
print(
    "Lower isovalues include more voxels and may capture additional specimen detail, "
    "but can also introduce more noise. "
    "Higher isovalues include fewer voxels and may reduce noise, "
    "but can remove real specimen structure."
)
print()
print(
    "We can help estimate a starting isovalue, or you can enter a value "
    "yourself."
)
print()

use_iso_helper = ask_yes_no(
    "Would you like help estimating a baseline isovalue?\n"
    "The helper will ask you to click several obvious background/air points\n"
    "and several obvious specimen/material points, then estimate the midpoint\n"
    "between their average grayscale values.\n"
    "Choose no if you already know the isovalue you want to test.",
    default="y"
)

if use_iso_helper:

    preview_image = create_iso_preview_image(...)

    iso, iso_helper_metadata = estimate_iso_from_click_samples(
        preview_image
    )

else:

    iso = ask(
        "What isovalue would you like to use for this surfacing run?",
        cast=float
    )

    iso_helper_metadata = None


# ============================================================
# Set padding
# ============================================================

padding = ask(
    "padding adds a small margin around each extracted specimen box.\n"
    "The unit is measured in voxels.",
    default=5,
    cast=int
)

# ============================================================
# Set parallelization
# ============================================================

default_extract_ncores = max(1, int(multiprocessing.cpu_count() * 0.85))
default_surface_ncores = min(20, multiprocessing.cpu_count())

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

# TODO(dev): This is old code for core use. Do we need to incorporate it or delete it?
# If we are incorporating it does it need updating? 

extract_num_cores = (
    extract_ncores
    if extract_ncores is not None
    else max(1, int(multiprocessing.cpu_count() * 0.85))
)

surface_num_cores = (
    surface_ncores
    if surface_ncores is not None
    else min(20, multiprocessing.cpu_count())
)

start = timeit.default_timer()

# ============================================================
# I'm not sure what this is
# ============================================================

# TODO(dev): This is code from before but perhaps is very important for the surfacing, so do we keep it and if so, I think we may need to update it. 

for t in range(tier_ids.max() + 1):
    infot = info[tier_ids == t, :].copy()
    zrng = tier_ranges[t]

    ang2rot2 = float(infot[0, 7])

    for i in range(zrng[0], min(zrng[1], len(fnames))):
        im = io.imread(os.path.join(slicepath, fnames[i]))
        im = apply_preview_orientation(im, transpose_preview)
        im = rotate(im, ang2rot, preserve_range=True)
        im = im[rowrng1[0]:rowrng1[1], colrng1[0]:colrng1[1]].copy()
        im = rotate(im, ang2rot2, preserve_range=True)

        if i == zrng[0]:
            infot[:, 3] = np.maximum(infot[:, 3] - PADDING, 0)
            infot[:, 5] = np.maximum(infot[:, 5] - PADDING, 0)
            infot[:, 4] = np.minimum(infot[:, 4] + PADDING, im.shape[0])
            infot[:, 6] = np.minimum(infot[:, 6] + PADDING, im.shape[1])

        Parallel(n_jobs=extract_num_cores)(
            delayed(extract_subvolume_slice)(i, j, infot, zrng, im, outpath) 
            for j in range(infot.shape[0])
        )

# ============================================================
# Save overview images and compressed voxel volumes
# ============================================================

# TODO(dev): This is old code and needs to be updated

for i in range(info.shape[0]):
    fname = os.path.join(outpath, info[i, 0])
    IMAGES = np.load(fname + ".npy", mmap_mode="r")
    overview = dicom.bone_overview(IMAGES)
    plt.imsave(fname + ".png", overview, cmap="gray")

    shape_out = IMAGES.shape
    np.savez_compressed(fname, I=IMAGES, dx=dx, dz=dz)
    del IMAGES
    os.remove(fname + ".npy")
    print("finished ", fname, " size: ", shape_out)

stop = timeit.default_timer()
print("subvol extraction runtime: ", stop - start)

print(transpose_preview)

# ============================================================
# Surfacing
# ============================================================

#TODO(dev): I think we need to keep this but I think it needs to be updated. 

print("starting DICOM surfacing")
dicom.surface_bones_parallel(
    outpath,
    iso=ISOLEVEL,
    write_gif=False,
    mirror=transpose_preview,
    ncores=surface_num_cores
)

# ============================================================
# Save metadata
# ============================================================



"iso": iso,
"iso_helper": iso_helper_metadata,


# ============================================================
# Confirmation and end --- next script stuff
# ============================================================



