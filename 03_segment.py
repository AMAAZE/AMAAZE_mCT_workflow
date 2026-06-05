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
# Configuration and imports
# ============================================================

from utils import *

# ============================================================
# Load metadata and subvolume
# ============================================================

print()
dataset_path = ask_existing_path(
    "What is the name of the dataset folder you want to continue working on?\n"
    "This should be the same dataset folder you gave to 00_share_data.py.",
    is_dir=True
)

metadata_path = find_metadata_file_in_dataset(dataset_path)
metadata = load_metadata_if_available(metadata_path)

scanpath = metadata["paths"]["scanpath"]
output_path = metadata["paths"]["output_path"]
layoutfile = metadata["paths"]["layoutfile"]

dataset_name = metadata["dataset_name"]
scan_num = metadata["scan_num"]

npz_fname = metadata["outputs"]["npz_file"]

if not os.path.exists(npz_fname):
    raise RuntimeError(
        f"Processed volume not found: {npz_fname}. "
        "Run 02_build_volume.py first."
    )

saveddata = np.load(npz_fname)

vol = saveddata["vol"]
rowrng = saveddata["rowrng"]
colrng = saveddata["colrng"]
rotation_angle = float(saveddata["ang"])
origsz = saveddata["origsz"]
rem = saveddata["remainder"]
transpose_preview = bool(saveddata["transpose_preview"])

rowsz = rowrng[1] - rowrng[0]
colsz = colrng[1] - colrng[0]

metadata["workflow"]["03_segment"]["inputs"] = {
    "npz_file": npz_fname,
    "volume_shape": list(vol.shape),
    "rowrng": rowrng.tolist(),
    "colrng": colrng.tolist(),
    "rotation_angle": float(rotation_angle),
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
# This step operates in reduced .npz space.
#
# Each Z slice is collapsed to one average intensity value, producing
# a 1D signal through the scan. Broad peaks in that signal correspond
# to major packaging boundaries, such as the box edges and tier dividers.
#
# Candidate peaks are detected with scipy.signal.find_peaks(), and
# prominence is used to choose the strongest internal tier boundaries
# expected from the layout CSV.

q = -np.mean(np.mean(vol,1),1)
tier_signal = q.copy()
q = q - q.min()

"""
NOTE(dev): 3e7 is a magic number. Review
"""
thresh = 3e7/(vol.shape[1]*vol.shape[2])
q[q<thresh] = 0

"""
NOTE(dev): width=10 is a magic number. Review
"""
vert_pks, peak_props = find_peaks(q, width=10, prominence=0)

suggested_tier_boundaries = select_tier_boundaries_by_prominence(
    q=q,
    peaks=vert_pks,
    peak_props=peak_props,
    n_tiers=n_tiers
)

fig = plt.figure()
plt.plot(np.arange(len(q)),tier_signal)
plt.xlabel('slice height (z)'); plt.ylabel('(-) average tier density'); plt.title('tier segmentation')

for vvv in suggested_tier_boundaries:
    plt.axvline(x=vvv, color='green', linestyle='--', linewidth=2)

print(
    "green lines are suggested tier boundaries. \n"
    "Click final tier boundary positions only if the suggestions are not correct. \n"
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
    ex = suggested_tier_boundaries
    tier_detection_method = "automatic_peaks"
    print("using ", ex)

tier_metadata = {}
metadata["workflow"]["03_segment"]["tiers"] = tier_metadata
tier_metadata["tier_boundaries"] = [int(v) for v in ex]
tier_metadata["n_detected_tiers"] = len(ex) - 1
tier_metadata["tier_detection_method"] = tier_detection_method

# Convert the selected tier boundary positions into start/end ranges.
# Each range is one tier in the reduced .npz volume.

ranges = [
    [ex[i], ex[i + 1]]
    for i in range(len(ex) - 1)
]

# Scan order and layout order are not always the same.
# If configured, reverse the detected tier order before matching
# tiers to the layout CSV.

tier_metadata["reverse_detected_tier_order"] = reverse_detected_tier_order

if reverse_detected_tier_order:
    ranges = ranges[::-1]

# Remove tiers that contain no specimens according to the layout file.

active_tiers = np.arange(n_tiers)[tier_mask]
ranges = [ranges[i] for i in active_tiers]

tier_metadata["active_tier_indices"] = [int(v) for v in active_tiers]
tier_metadata["active_tier_ranges"] = [
    [int(start), int(end)] for start, end in ranges
]

# Extract the reduced-volume data for each active tier.
# Downstream divider and cell segmentation operate in .npz space.

SLICES = [
    vol[start:end, :, :]
    for start, end in ranges
]

# Mean projection of each active tier used for divider detection.
I = [x.mean(0) for x in SLICES]

# Final extraction instructions.
# Written to CT<scan_num>.csv and used by 04_surface.py.

EXTRACTS = []


# ============================================================
# Per-tier geometric normalization via rotation
# ============================================================

# This step applies a small per-tier rotation correction before 
# divider detection. The applied rotation angle is recorded in
# run metadata for replication.

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
    
    normalized_tier_images.append(normalized_image)

    rotation_metadata.append({
        "tier_index": int(i),
        "rotation_angle": best_angle,
        "angle_min": -5,
        "angle_max": 5,
        "angle_step": 0.25,
        "best_score": float(rotation_scores.max()),
    })
    
metadata["workflow"]["03_segment"]["tier_rotations"] = rotation_metadata
    
# ============================================================
# Per-tier divider detection
# ============================================================

"""
Divider detection operates on the mean projection of each active tier.

A local stability image is computed to emphasize specimen and divider
boundaries while suppressing much of the internal texture. The stability
image is thresholded and converted into row- and column-wise occupancy
profiles.

Rows and columns with high occupancy are treated as candidate divider
edges. Neighboring candidate indices are collapsed into edge bands,
paired according to plausible divider widths, and converted into divider
centerlines.

Diagnostic plots are generated to visualize:
1. The stability image.
2. Candidate divider edges.
3. Paired divider edges.
4. Final divider centerlines.

The resulting divider network is used to define specimen extraction
regions for downstream processing.
"""

divider_proposals = []

for i, normalized_image in enumerate(normalized_tier_images):

    """
    NOTE(dev): window_size=3, percentile_low=2,
    and percentile_high=98 heuristic parameters.

    Smaller windows produce sharper local edge responses.
    Larger windows broaden the stability halos and may merge
    nearby features.

    Values below the 2nd percentile and above the 98th percentile
    are clipped before normalization to reduce the influence of
    extreme intensity values.

    Current value selected empirically because it produced
    stable divider detection on test datasets while preserving
    boundary localization.

    Review across additional datasets.
    """
    stability_window = 3

    stability_image = local_stability_image(
        normalized_image,
        window_size=stability_window,
        percentile_low=2,
        percentile_high=98
    )

    """
    NOTE(dev): dark-mask percentile cutoff = 20 is a heuristic parameter.

    Occupancy responses below this value are discarded prior to
    candidate divider selection.
    
    Occupancy is the number of pixels within a row or column that
    are classified as stability features after thresholding the
    local stability image.

    Lower values increase sensitivity but may introduce weak or
    spurious divider candidates. Higher values suppress weak
    responses and favor stronger divider signals.

    Current value selected empirically during divider-detection
    development. Review across additional datasets.
    """

    cutoff = np.percentile(stability_image, 20)
    dark_mask = stability_image < cutoff

    """
    NOTE(dev): occupancy_threshold=0.45, min_pair_gap=20,
    and max_pair_gap=20 are heuristic parameters.

    occupancy_threshold defines the minimum fraction of pixels
    classified as stability features required for a row or column
    to become a candidate divider edge.

    min_pair_gap and max_pair_gap define the allowable separation
    between candidate edge bands when pairing opposite sides of a
    divider.

    Lower occupancy thresholds increase sensitivity but may
    introduce false positives. Wider pairing ranges permit more
    divider-width variation but may increase incorrect pairings.

    Current values were selected empirically during development.
    Review across additional datasets.
    """

    row_centers, row_pairs, row_occupancy, row_candidates, row_candidate_bands, row_band_centers = paired_edge_centerlines(
        dark_mask,
        axis_label="row",
        min_fraction=0.45,
        max_pair_gap=20
    )

    col_centers, col_pairs, col_occupancy, col_candidates, col_candidate_bands, col_band_centers = paired_edge_centerlines(
        dark_mask,
        axis_label="col",
        min_fraction=0.45,
        max_pair_gap=20
    )

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Stability image
    axs[0, 0].imshow(stability_image, cmap="gray")
    axs[0, 0].set_title(f"Tier {i+1}: stability image")
    axs[0, 0].axis("off")

    # 2. Candidate divider edges
    axs[0, 1].imshow(normalized_image, cmap="gray")

    for r in row_candidates:
        axs[0, 1].axhline(r, color="magenta", linestyle=":", linewidth=1)

    for c in col_candidates:
        axs[0, 1].axvline(c, color="magenta", linestyle=":", linewidth=1)

    axs[0, 1].set_title("Candidate divider edges")
    axs[0, 1].axis("off")

    # 3. Paired divider edges
    axs[1, 0].imshow(normalized_image, cmap="gray")

    for r1, r2 in row_pairs:
        axs[1, 0].axhline(r1, color="red", linestyle="--")
        axs[1, 0].axhline(r2, color="green", linestyle="--")

    for c1, c2 in col_pairs:
        axs[1, 0].axvline(c1, color="red", linestyle="--")
        axs[1, 0].axvline(c2, color="green", linestyle="--")

    axs[1, 0].set_title("Paired divider edges")
    axs[1, 0].axis("off")

    # 4. Divider centerlines
    axs[1, 1].imshow(normalized_image, cmap="gray")

    for r1, r2 in row_pairs:
        axs[1, 1].axhline(r1, color="red", linestyle="--")
        axs[1, 1].axhline(r2, color="green", linestyle="--")

    for r in row_centers:
        axs[1, 1].axhline(r, color="cyan", linewidth=2)

    for c1, c2 in col_pairs:
        axs[1, 1].axvline(c1, color="red", linestyle="--")
        axs[1, 1].axvline(c2, color="green", linestyle="--")

    for c in col_centers:
        axs[1, 1].axvline(c, color="cyan", linewidth=2)

    axs[1, 1].set_title("Divider centerlines")
    axs[1, 1].axis("off")

    fig.suptitle(f"Tier {i+1}: automated divider proposal", fontsize=14)
    plt.tight_layout()
    plt.show()
       
    review_choice = input(
        "\nPress ENTER to accept automated dividers "
        "or type 'm' for manual override: "
    )

    divider_proposals.append({
        "tier_id": int(tier_ids[i]),
        "proposed_rows": np.array(row_centers).astype(int),
        "proposed_cols": np.array(col_centers).astype(int),
        "review_choice": review_choice.strip().lower()
    })

# ============================================================
# Divider review / correction layer
# ============================================================
# manual-only mode: proposals are empty
# assisted mode: proposals come from automation

final_divider_metadata = []

for i, normalized_image in enumerate(normalized_tier_images):

    tier_id = int(tier_ids[i])

    proposed_rows = divider_proposals[i]["proposed_rows"]
    proposed_cols = divider_proposals[i]["proposed_cols"]

    if divider_proposals[i]["review_choice"] == "m":

        final_rows, final_cols = review_dividers(
            image=normalized_image,
            title=f"Tier {tier_id}: divider review"
        )

    else:

        final_rows = proposed_rows
        final_cols = proposed_cols
    
    final_divider_metadata.append({
        "tier_id": tier_id,
        "proposed_row_dividers": proposed_rows.tolist(),
        "proposed_col_dividers": proposed_cols.tolist(),
        "final_row_dividers": final_rows.tolist(),
        "final_col_dividers": final_cols.tolist(),
        "divider_method": (
            "automatic_accepted"
            if np.array_equal(final_rows, proposed_rows)
            and np.array_equal(final_cols, proposed_cols)
            else "manual_override"
        ),
    })

metadata["workflow"]["03_segment"]["final_dividers"] = final_divider_metadata


# ============================================================
# Future work: final extraction handoff
# ============================================================

save_metadata(metadata_path, metadata)

"""
NOTE (dev)
write the tier/grid/divider information in the exact form needed by downstream scripts, either preserving the old expected format or creating a cleaner format plus compatibility export.
"""

# ============================================================
# Update metadata
# ============================================================

metadata["workflow"]["03_segment"] = {
    "status": "complete",
    "inputs": {
        "npz_file": npz_fname,
        "volume_shape": list(vol.shape),
        "rowrng": rowrng.tolist(),
        "colrng": colrng.tolist(),
        "rotation_angle": float(rotation_angle),
        "transpose_preview": transpose_preview,
    },
    "tiers": tier_metadata,
    "tier_rotations": rotation_metadata,
    "final_dividers": final_divider_metadata,
}

save_metadata(metadata_path, metadata)

# ============================================================
# Confirm completion
# ============================================================
print()
print("Subvolume created.")
print("Metadata updated:")
print(metadata_path)
print()
print("Next step:")
print("python 04_surface.py")

