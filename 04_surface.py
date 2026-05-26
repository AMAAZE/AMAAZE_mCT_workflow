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

from utils import *

scanpath = os.path.normpath(scanpath)
slicepath = os.path.normpath(slicepath)

if not os.path.isdir(scanpath):
    raise RuntimeError("The provided scanpath does not exist or is not a folder.")

if not os.path.isdir(slicepath):
    raise RuntimeError("The provided slicepath does not exist or is not a folder.")

if calibration_filename is not None:
    calibration_fname = os.path.join(scanpath, calibration_filename)
else:
    calibration_fname = None

csvpath = os.path.join(scanpath, f"CT{scan_num}.csv")
npz_fname = os.path.join(scanpath, f"ct{scan_num}_new.npz")

if not os.path.exists(csvpath):
    raise RuntimeError(f"Extraction CSV not found: {csvpath}.")

if not os.path.exists(npz_fname):
    raise RuntimeError(f"Processed volume not found: {npz_fname}.")

outpath = os.path.join(scanpath, "Meshes")
os.makedirs(outpath, exist_ok=True)

saveddata = np.load(npz_fname)
info = pd.read_csv(csvpath, header=None).to_numpy()

fnames = [f for f in os.listdir(slicepath) if f.lower().endswith('.tif')]

# Sort using numeric index (compute once)
fnames_with_idx = [(f, extract_index(f)) for f in fnames]
fnames_with_idx.sort(key=lambda x: x[1])

fnames = [f for f, _ in fnames_with_idx]
indices = [idx for _, idx in fnames_with_idx]

if len(fnames) == 0:
    raise RuntimeError("No .tif files found in the provided slicepath.")

# Resolve voxel size in mm.
# Priority:
# 1) voxel_size_mm from user_inputs.json
# 2) calibration report
# 3) error if neither is available
dx = get_voxel_size_mm(
    calibration_fname=calibration_fname,
    user_voxel_size_mm=voxel_size_mm
)

if voxel_spacing_mm is None:
    dz = dx
    print(f"No voxel_spacing_mm provided; using isotropic spacing dz = dx = {dx} mm")
else:
    dz = voxel_spacing_mm
    print(f"Using voxel_spacing_mm from user_inputs.json: dz = {dz} mm")

tier_ranges, tier_ids = np.unique(info[:, 1:3].astype(int), axis=0, return_inverse=True)

rowrng1 = saveddata["rowrng"]
colrng1 = saveddata["colrng"]
ang2rot = saveddata["ang"]
origsz = saveddata["origsz"]
rem = saveddata["remainder"]
transpose_preview = bool(saveddata["transpose_preview"])

PADDING = padding
ISOLEVEL = iso

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

# Save overview images and compressed voxel volumes.
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
print("starting DICOM surfacing")
dicom.surface_bones_parallel(
    outpath,
    iso=ISOLEVEL,
    write_gif=False,
    mirror=transpose_preview,
    ncores=surface_num_cores
)


