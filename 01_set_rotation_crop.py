#!/usr/bin/env python3

"""
01_set_rotation_crop.py

Original processing logic: RileyWilde
Refactoring and workflow design: Katrina E. Yezzi-Woodley

Preview a representative CT slice using the current user settings,
apply preview orientation, rotation, and crop, and write the selected
rotation/crop settings to controls.txt for downstream processing.
"""

from utils import *


if slice_index_fraction is None:
    raise RuntimeError("'slice_index_fraction' required in user inputs .json file")

if ang2rot is None:
    raise RuntimeError("'ang2rot' required in user_inputs.json")

if rowrng is None or len(rowrng) != 2:
    raise RuntimeError("'rowrng' required as [start, end] in user inputs .json file")

if colrng is None or len(colrng) != 2:
    raise RuntimeError("'colrng' required as [start, end] in user inputs .json file")

if not isinstance(transpose_preview, bool):
    raise RuntimeError("'transpose_preview' must be True or False in user inputs .json file")

# TODO(dev): Generalize input discovery beyond TIFF stacks.
# Current implementation supports .tif/.tiff image stacks only.
# Future versions should handle DICOM and other scan formats through
# an explicit input-loader layer rather than adding ad hoc glob patterns here.

slicepath = os.path.normpath(slicepath)

tif_files = []
for ext in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
    tif_files.extend(glob.glob(os.path.join(slicepath, ext)))

tif_files = sorted(set(tif_files))

# Stop early if the slice folder is wrong or contains no TIFF slices.
if len(tif_files) == 0:
    raise RuntimeError("No .tif or .tiff files found in the specified slice folder.")

# Sort by numeric slice index rather than filename text so non-padded names
# do not scramble slice order.
tif_files_with_idx = [(f, extract_index(f)) for f in tif_files]
tif_files_with_idx.sort(key=lambda x: x[1])

tif_files = [f for f, _ in tif_files_with_idx]
indices   = [idx for _, idx in tif_files_with_idx]

if any(indices[i] >= indices[i+1] for i in range(len(indices)-1)):
    raise RuntimeError("Slice files are not strictly increasing after sorting.")

# Check for missing slice indices; allow warning-only behavior if configured.
expected = list(range(indices[0], indices[0] + len(indices)))

if indices != expected:
    msg = "Slice indices are not consecutive. Missing slices detected in slicepath."
    
    if allow_slice_gaps:
        print("Warning:", msg)
    else:
        raise RuntimeError(msg)

print("First 5 slices:", [os.path.basename(f) for f in tif_files[:5]])
print("Last 5 slices:", [os.path.basename(f) for f in tif_files[-5:]])

# Choose one representative slice for preview based on the user-provided fraction.
slice_index = int(len(tif_files) * slice_index_fraction)
slice_index = min(slice_index, len(tif_files) - 1)
slice_file = tif_files[slice_index]
    
I = plt.imread(slice_file)

# Apply preview-only orientation before rotation/cropping for visual alignment.
# NOTE(dev): Transform order matters.
# 1. transpose_preview aligns TIFF-native orientation with layout orientation.
# 2. ang2rot rotates the layout-aligned image for display/segmentation.
# 3. resize=True preserves the full rotated canvas for rectangular scans.
# 4. rowrng/colrng crop the transformed display-space image.
imdisp = apply_preview_orientation(I, transpose_preview) 
imdisp = rotate(imdisp, ang2rot, preserve_range=True, resize=True)
imdisp = imdisp[rowrng[0]:rowrng[1], colrng[0]:colrng[1]].copy()

# TODO(dev): Replace manual JSON crop bounds with click-based crop selection.
# For now, rowrng and colrng are entered in user_inputs.json.
# These coordinates are in display space after transpose_preview and ang2rot,
# not in raw TIFF-native coordinate space.

# Visual check: adjust JSON settings and rerun if needed.  
plt.imshow(imdisp)
plt.title(f"Preview slice | rotation={ang2rot}, rows={rowrng}, cols={colrng}")

# This file will be saved in the scan directory
controls_fname = os.path.join(scanpath, "controls.txt")

# NOTE: Output format used by downstream scripts.
# Revisit if modifying key names, spacing, or delimiters.
with open(controls_fname, "w") as f:
    f.write(f"ang2rot: {ang2rot}\n")
    f.write(f"rowrng: {rowrng}\n")
    f.write(f"colrng: {colrng}\n")
    f.write(f"transpose_preview: {transpose_preview}\n")

print(f"Rotation and crop settings saved to {controls_fname}")
plt.show()
