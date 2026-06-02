"""
03_segment.py

Original processing logic: RileyWilde
Refactoring and workflow design: Katrina E. Yezzi-Woodley

Load the processed volume and segment it into tiers and specimen regions
using manual divider definition informed by the scan layout.

This step:
- determines vertical tier boundaries,
- make minor per-tier rotations for normalization,
- suggested row/column divider locations based on automated detection,
- allows the user to manually define row/column dividers overrides per tier,
- computes specimen extraction boxes,
- and saves the extraction plan to CT<scan_num>.csv
for downstream extraction and surfacing.

"""

# ============================================================
# Validate required input paths
# ============================================================

from utils import *

slicepath = os.path.normpath(slicepath)

if not os.path.isdir(slicepath):
    raise RuntimeError("The provided slicepath was not found.")

if not os.path.exists(layoutfile): 
    raise RuntimeError("The layoutfile was not found.")

# ============================================================
# Load workflow configuration from JSON
# ============================================================
# User-authored configuration is loaded through utils.py.
#
# This script should not read or write ct<scan_num>_params.txt.
# The old params.txt file mixed configuration, cached results,
# remembered user choices, and provenance.
#
# In this refactor:
# - JSON stores user-provided configuration.
# - Runtime metadata objects store values generated during this run.
# - A future run-metadata export will preserve provenance after execution.

# ============================================================
# Load run metadata
# ============================================================

run_metadata = load_run_metadata(scanpath, scan_num)

run_metadata["workflow"]["03_segment"] = {}


# ============================================================
# Load TIFF filenames
# ============================================================

"""
NOTE (dev)
Do we need this if we are going to use only npz space?
"""

# ============================================================
# Load reduced working volume
# ============================================================

npz_fname = os.path.join(scanpath, f"ct{scan_num}_new.npz")

if not os.path.exists(npz_fname):
    raise RuntimeError(
        f"Processed volume not found: {npz_fname}. "
        "Run 02_build_volume.py first."
    )

saveddata = np.load(npz_fname)

vol = saveddata["vol"]
rowrng = saveddata["rowrng"]
colrng = saveddata["colrng"]
ang2rot = saveddata["ang"]
origsz = saveddata["origsz"]
rem = saveddata["remainder"]
transpose_preview = bool(saveddata["transpose_preview"])

rowsz = rowrng[1] - rowrng[0]
colsz = colrng[1] - colrng[0]

run_metadata["workflow"]["03_segment"]["inputs"] = {
    "npz_file": npz_fname,
    "volume_shape": list(vol.shape),
    "rowrng": rowrng.tolist(),
    "colrng": colrng.tolist(),
    "ang2rot": float(ang2rot),
    "transpose_preview": transpose_preview,
}

# ============================================================
# Load scan layout CSV
# ============================================================
# The layout CSV tells the workflow what specimens are supposed to be
# in each cell of the packaging grid.
#
# This does NOT detect dividers. It only tells the workflow:
# - how many tiers are expected
# - how many rows/columns are expected
# - what each extracted cell should be named

# Empty cells are converted to 0 by fillna(0), and later ignored.
x = pd.read_csv(layoutfile).fillna(0)
x = x.to_numpy()

# Keep only the rows for the scan currently being processed.
# Column 0 is scan number, so after filtering we remove it with [:, 1:].
x = x[x[:,0]==scan_num,1:].copy()
    
if len(x) == 0:
    raise ValueError(
        f'Scan {scan_num} not found in layout file. '
        'Check that the CSV includes this scan and follows the required format.'
    )

# Number of tiers listed in the layout file.
n_tiers = int(x[:,0].max())

layout_by_tier = {}

for tier_id in range(1, n_tiers + 1):
    tier_data = x[x[:, 0] == tier_id]

    if len(tier_data) == 0:
        continue

    n_rows = int(tier_data[:, 1].max())
    n_cols = tier_data.shape[1] - 2

    tier_layout = np.zeros((n_rows, n_cols), object)

    for row_id in range(1, n_rows + 1):
        row_data = tier_data[tier_data[:, 1] == row_id, 2:]

        if row_data.shape[0] != 1:
            raise ValueError(
                f"Tier {tier_id}, row {row_id} has {row_data.shape[0]} matching rows. "
                "Expected exactly one."
            )

        tier_layout[row_id - 1, :] = row_data[0]

    layout_by_tier[tier_id] = {
        "layout": tier_layout,
        "mask": tier_layout != 0,
        "n_rows": n_rows,
        "n_cols": n_cols,
    }

tier_ids = np.array(sorted(layout_by_tier.keys()))
tier_mask = np.array([np.any(layout_by_tier[t]["mask"]) for t in tier_ids])


# ============================================================
# Tier Division
# ============================================================

"""
NOTE (dev)
Detect or define tiers, show user-verifiable output, record tier boundaries explicitly.
Good code exists for this. 
I would like to compare the code for this on the main branch to the code for this on the second branch.
This already has a user-integrated step whereby they can click on the grid to set the boundaries.
"""

# This step works from the reduced .npz volume, not directly from the original TIFF stack.
#
# It collapses each Z slice into one average value, producing a 1D signal
# through the stack. Peaks/dips in that signal are used to identify tier
# boundaries.

q = -np.mean(np.mean(vol,1),1)
sig = q.copy()
q = q - q.min()

thresh = 3e7/(vol.shape[1]*vol.shape[2])
q[q<thresh] = 0

# NOTE: peak width controls minimum feature size detected in the 1D projection.
# If divider contrast is low or features are narrow, this may need adjustment.
# This is a heuristic parameter, not a fixed physical constant.
vert_pks = find_peaks(q,width=10)[0]
vert_pks = np.concatenate((np.array([0]),vert_pks))
vert_pks = np.concatenate((vert_pks,np.array([len(q)])) )

fig = plt.figure()
plt.plot(np.arange(len(q)),sig)
plt.xlabel('slice height (z)'); plt.ylabel('(-) average tier density'); plt.title('tier segmentation')
for vvv in vert_pks:
    plt.axvline(x=vvv, color='green', linestyle='--', linewidth=2)

print(
    "green lines are candidate vertical tier-boundary peaks. \n"
    "Click the final tier boundary positions if the candidates are not correct. \n"
    "# tiers is %1d, # nonempty tiers is %2d \n" % (n_tiers, np.sum(tier_mask))
)
        
clicked_x = []  # store clicked x-values

def onclick(event):
    if event.inaxes:
        x_click = event.xdata
        clicked_x.append(x_click)
        # Draw vertical line
        event.inaxes.axvline(x_click, color="r", linestyle="--")
        plt.draw()
        print(f"Clicked x = {x_click:.2f}")

cid = fig.canvas.mpl_connect("button_press_event", onclick)

plt.show(block=False)        
        
input("please press enter once done (no clicks = use suggested values) \n")

if len(clicked_x) > 0:
    ex = np.array(clicked_x).astype(int)
    ex[ex<0] = 0
    ex[ex>len(q)] = len(q)
    tier_detection_method = "manual_override"
    print("new vertical peaks", ex)
else:
    ex = vert_pks
    tier_detection_method = "automatic_peaks"
    print("using ", ex)

tier_metadata = {}
run_metadata["workflow"]["03_segment"]["tiers"] = tier_metadata
tier_metadata["tier_boundaries"] = [int(v) for v in ex]
tier_metadata["n_detected_tiers"] = len(ex) - 1
tier_metadata["tier_detection_method"] = tier_detection_method

# Convert the selected tier boundary positions into start/end ranges.
# Each range is one tier in the reduced .npz volume.
#
# REVIEW:
# The ranges are reversed here. This may be intentional because of scan/layout
# orientation, but we should confirm before changing it.

ranges = []
for i in range(len(ex)-1):
    ranges.append([ex[i],ex[i+1]])

# TEMP(dev): Explicitly recording current tier-order behavior.
# Replace with JSON-configured tier order once that setting is added.
tier_metadata["reverse_detected_tier_order"] = True

if tier_metadata["reverse_detected_tier_order"]:
    ranges = ranges[-1::-1]

tiers = np.arange(n_tiers)[tier_mask]
ranges = [ranges[i] for i in tiers]

tier_metadata["active_tier_indices"] = [int(v) for v in tiers]
tier_metadata["active_tier_ranges"] = [
    [int(start), int(end)] for start, end in ranges
]

# Pull out the reduced-volume data for each tier.
# Each item in SLICES is one tier-sized chunk of the .npz volume.
#
# This means the downstream divider/cell segmentation is operating on
# reduced .npz data, not directly on the full TIFF stack.

SLICES = []
for i in range(len(ranges)):
    SLICES.append(vol[ranges[i][0]:ranges[i][1],:,:])

I = [x.mean(0) for x in SLICES]

# This will store the final extraction instructions.
# At the end of the script, these become CT<scan_num>.csv.
#
# That CSV is used by 04_surface.py to go back to the original TIFF stack and
# extract each specimen.

EXTRACTS = []




# ============================================================
# Per-tier geometric normalization via rotation
# ============================================================

normalized_tier_images = []
rotation_metadata = []

for i in range(len(I)):
    tier_image = I[i]

    best_angle, tested_angles, rotation_scores = estimate_grid_rotation_by_coherence(
        tier_image,
        angle_min=-5,
        angle_max=5,
        angle_step=0.25
    )

    normalized_image = rotate(
        tier_image,
        best_angle,
        preserve_range=True,
        resize=False,
        mode="edge"
    )

# Temp addition to build autoseg

    stability_image = local_stability_image(
        normalized_image,
        window_size=5
    )

    plt.figure()
    plt.imshow(stability_image, cmap="gray")
    plt.title(f"Tier {i+1}: local stability")
    plt.axis("off")
    plt.show()

# end of temp addition

    normalized_tier_images.append(normalized_image)

    rotation_metadata.append({
        "tier_index": int(i),
        "rotation_angle": best_angle,
        "angle_min": -5,
        "angle_max": 5,
        "angle_step": 0.25,
        "best_score": float(rotation_scores.max()),
    })

    plt.figure()
    plt.plot(tested_angles, rotation_scores)
    plt.axvline(
        best_angle, 
        linestyle="--",
        label=f"Best angle = {best_angle:.2f}°"
    )
    plt.title(
        f"Tier {i+1}: rotation coherence score\n"
        f"Best angle = {best_angle:.2f}°"
    )
    plt.xlabel("Rotation angle")
    plt.ylabel("Coherence score")
    plt.show()

    plt.figure()
    plt.imshow(normalized_image, cmap="gray")
    plt.title(f"Tier {i+1}: normalized image, angle={best_angle:.2f}")
    plt.axis("off")
    plt.show()

# input("Press Enter after reviewing rotation plots...")

run_metadata["workflow"]["03_segment"]["tier_rotations"] = rotation_metadata


# ============================================================
# Per-tier divider detection
# ============================================================
"""
NOTE (dev)
horizontal aggregation first, vertical aggregation second, with detected divider lines saved and previewed.
This is where we will try something new. 
For each geometrically normalized tier in NPZ space, we will generate representative 2D slice views or projections and compute horizontal and vertical intensity aggregation profiles across rows and columns. Divider detection will identify long, continuous, low-variance bright bands by evaluating changes in aggregated intensity and continuity across neighboring rows and columns rather than relying solely on individual voxel threshold values. Automatically detected divider boundaries will then be visualized for user verification, with manual click-based overrides available when needed before the finalized divider layout is passed downstream.
"""

divider_metadata = []

for i, normalized_image in enumerate(normalized_tier_images):

    tier_id = int(tier_ids[i])
    layouti = layout_by_tier[tier_id]["layout"]
    maski = layout_by_tier[tier_id]["mask"]

    row_profile = get_axis_low_variance_profile(
        normalized_image,
        axis_label="row",
        window_size=5
    )

    col_profile = get_axis_low_variance_profile(
        normalized_image,
        axis_label="col",
        window_size=5
    )

    row_bands, row_cutoff, row_bright = detect_bright_bands_from_profile(
        row_profile,
        threshold_fraction=0.85,
        min_band_width=2
    )

    col_bands, col_cutoff, col_bright = detect_bright_bands_from_profile(
        col_profile,
        threshold_fraction=0.85,
        min_band_width=2
    )

    divider_metadata.append({
        "tier_index": int(i),
        "tier_id": tier_id,
        "layout_shape": list(layouti.shape),
        "row_bands": row_bands,
        "col_bands": col_bands,
        "row_profile_cutoff": float(row_cutoff),
        "col_profile_cutoff": float(col_cutoff),
    })

    plt.figure()
    plt.plot(row_profile)
    plt.axhline(row_cutoff, linestyle="--")
    for start, end in row_bands:
        plt.axvspan(start, end, alpha=0.25)
    plt.title(f"Tier {tier_id}: horizontal divider-band profile")
    plt.xlabel("Image row")
    plt.ylabel("Continuity score")
    plt.show()

    plt.figure()
    plt.plot(col_profile)
    plt.axhline(col_cutoff, linestyle="--")
    for start, end in col_bands:
        plt.axvspan(start, end, alpha=0.25)
    plt.title(f"Tier {tier_id}: vertical divider-band profile")
    plt.xlabel("Image column")
    plt.ylabel("Continuity score")
    plt.show()

    plt.figure()
    plt.imshow(normalized_image, cmap="gray")
    for start, end in row_bands:
        plt.axhspan(start, end, alpha=0.25)
    for start, end in col_bands:
        plt.axvspan(start, end, alpha=0.25)
    plt.title(f"Tier {tier_id}: proposed divider bands")
    plt.axis("off")
    plt.show()

run_metadata["workflow"]["03_segment"]["divider_detection"] = divider_metadata


# ============================================================
# Divider review / correction layer
# ============================================================

final_divider_metadata = []

for i, normalized_image in enumerate(normalized_tier_images):

    tier_id = int(tier_ids[i])

    proposed_rows = []
    proposed_cols = []

    final_rows, final_cols, nondivider_points = review_dividers(
        image=normalized_image,
        proposed_rows=proposed_rows,
        proposed_cols=proposed_cols,
        title=f"Tier {tier_id}: divider review"
    )

    final_divider_metadata.append({
        "tier_id": tier_id,
        "final_row_dividers": final_rows.tolist(),
        "final_col_dividers": final_cols.tolist(),
        "nondivider_points": nondivider_points,
    })

run_metadata["workflow"]["03_segment"]["final_dividers"] = final_divider_metadata

save_run_metadata(scanpath, scan_num, run_metadata)

# ============================================================
# Temporary Experiment to try and find the right parameter for autodetect
# ============================================================

print("Final rows:", final_rows)
print("Final cols:", final_cols)
print("Non-divider points:", nondivider_points)

for row, col in nondivider_points:

    divider_row = final_rows[0]

    plt.figure()

    plt.plot(
        normalized_image[divider_row, :],
        label=f"Divider row {divider_row}"
    )

    plt.plot(
        normalized_image[row, :],
        label=f"Non-divider row {row}"
    )

    plt.title("Divider row vs non-divider row")
    plt.xlabel("Column")
    plt.ylabel("Intensity")
    plt.legend()

    plt.show()

    divider_col_1 = final_cols[0]

    plt.figure()

    plt.plot(
        normalized_image[:, divider_col_1],
        label=f"Divider column {divider_col_1}"
    )

    plt.plot(
        normalized_image[:, col],
        label=f"Non-divider column {col}"
    )

    plt.title("Divider column vs non-divider column")
    plt.xlabel("Row")
    plt.ylabel("Intensity")
    plt.legend()

    plt.show()

# ============================================================
# Final layout output
# ============================================================

save_run_metadata(scanpath, scan_num, run_metadata)

"""
NOTE (dev)
write the tier/grid/divider information in the exact form needed by downstream scripts, either preserving the old expected format or creating a cleaner format plus compatibility export.
"""
