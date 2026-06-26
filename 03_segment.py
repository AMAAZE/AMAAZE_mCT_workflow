"""
03_segment_backup.py

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
- and saves the extraction plan to a .csv
for downstream extraction and surfacing.

"""

# ============================================================
# Configuration and imports
# ============================================================

from utils import *

# ============================================================
# Load metadata
# ============================================================

timer_03_start = timeit.default_timer()

#print()
#dataset_path = ask_existing_path(
#    "What is the full path to the dataset folder you want to continue working on?\n"
#    "This should be the same dataset folder path you gave to 00_share_data.py.\n"
#    "Example:\n"
#    "C:/MyProject/CT_scan_01",
#    is_dir=True
#)

#metadata_path = find_metadata_file_in_dataset(dataset_path)
#metadata = load_metadata_if_available(metadata_path)


metadata_paths = get_metadata_paths_from_command_line_or_user(
    step_name="03_segment",
    allow_batch=False
)

metadata_path = metadata_paths[0]
metadata = load_metadata_if_available(metadata_path)

(
    dataset_folder_name,
    scanpath,
    slicepath,
    layoutfile,
    output_path,
    metadata_path,
    slice_index_fraction,
    voxel_size_mm,
    voxel_spacing_mm,
    transpose_preview,
    rotation_angle,
    rowrng,
    colrng,
    subvolume_file,
) = unpack_metadata(metadata)

# ============================================================
# Load subvolume (npz) from 02
# ============================================================

subvolume_file = metadata["02_build_subvolume"]["subvolume_file"]

if not os.path.exists(subvolume_file):
    raise RuntimeError(
        f"Processed volume not found: {subvolume_file}. "
        "Run 02_build_subvolume.py first."
    )

npzdata = np.load(subvolume_file)

# Extract data for this workflow from npz
vol = npzdata["vol"]

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

# Empty cells are converted to 0 by fillna(0) and are treated as
# intentionally empty positions within the specimen layout.
x = pd.read_csv(layoutfile).fillna(0)
x = x.to_numpy()

# The layout CSV describes only the current dataset.
# Column 0 is the tier number.
if len(x) == 0:
    raise ValueError(
        "The layout CSV appears to be empty."
        "Check that the CSV describes the current dataset and follows the required format."
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

# tier_ids are the tier numbers present in the csv layout.
# tier_mask marks which of those tiers contain at least one specimen.
tier_ids = np.array(sorted(layout_by_tier.keys()))
tier_mask = np.array([np.any(layout_by_tier[t]["mask"]) for t in tier_ids])

# ============================================================
# Prepare diagnostic figure output folder
# ============================================================

diagnostic_figures_path = os.path.join(output_path, "03_diagnostic_figures")
os.makedirs(diagnostic_figures_path, exist_ok=True)

peak_diagnostics_csv = os.path.join(
    output_path,
    f"{dataset_folder_name}_peak_diagnostics.csv"
)

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

mean_intensity_profile_z = -np.mean(np.mean(vol, axis=1), axis=1)

shifted_mean_intensity_profile_z = (
    mean_intensity_profile_z
    - mean_intensity_profile_z.min()
)

vert_pks, peak_props, boundary_scores = generate_tier_boundary_candidates(
    shifted_mean_intensity_profile_z,
    n_tiers
)

tier_boundary_result = select_tier_boundaries_by_edge_and_score(
    shifted_mean_intensity_profile_z,
    candidate_peaks=vert_pks,
    candidate_peak_props=peak_props,
    candidate_boundary_scores=boundary_scores,
    n_tiers=n_tiers,
)

suggested_tier_boundaries = tier_boundary_result["selected_boundaries"]
left_edge = tier_boundary_result["left_edge"]
right_edge = tier_boundary_result["right_edge"]
suggested_internal = tier_boundary_result["suggested_internal"]

z_reduced = np.arange(len(mean_intensity_profile_z))

fig_tiers, axs = plt.subplots(1, 2, figsize=(16, 4), sharey=True)

axs[0].plot(
    z_reduced,
    shifted_mean_intensity_profile_z,
)

axs[0].plot(
    vert_pks,
    shifted_mean_intensity_profile_z[vert_pks],
    "ro",
    label="candidate peaks",
)

axs[0].axvline(
    left_edge,
    color="purple",
    linestyle="--",
    linewidth=2,
    label="detected edges",
)

axs[0].axvline(
    right_edge,
    color="purple",
    linestyle="--",
    linewidth=2,
)

axs[0].set_xlabel("slice height (z)")
axs[0].set_ylabel("shifted_mean_intensity_profile_z")
axs[0].set_title("Candidate peaks and independently detected edges")
axs[0].legend()

axs[1].plot(z_reduced, shifted_mean_intensity_profile_z)

axs[1].axvline(left_edge, color="purple", linestyle="--", linewidth=2, label="detected edges")
axs[1].axvline(right_edge, color="purple", linestyle="--", linewidth=2)

for vvv in suggested_internal:
    axs[1].axvline(vvv, color="green", linestyle="--", linewidth=2, label="selected internal divider")

handles, labels = axs[1].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
axs[1].legend(by_label.values(), by_label.keys())

axs[1].set_xlabel("slice height (z)")
axs[1].set_title("Suggested tier boundaries (edge-aware)")

fig_tiers.suptitle(
    f"{dataset_folder_name} | Tier segmentation review\n"
    f"(z-window: {metadata['02_build_subvolume']['zwindow']}, "
    f"original slices: {metadata['00_share_data']['n_slices']})" 
)

fig_tiers.tight_layout(rect=[0, 0, 1, 0.88])

tiers_png = os.path.join(
    diagnostic_figures_path,
    f"{dataset_folder_name}_tier_segmentation_review.png"
)

fig_tiers.savefig(
    tiers_png,
    dpi=300,
    bbox_inches="tight"
)

print(
    "\nA tier-boundary review window is opening.\n"
    "Purple dashed lines show the independently detected package edges.\n"
    "Green dashed lines show the selected internal tier boundaries.\n"
    "Together they form the suggested tier boundaries.\n"
    "If the suggestions look correct, do not click anything.\n"
    "If they are incorrect, click the graph where the tier boundaries should be.\n"
    "Each click will create a red divider line.\n"
    f"Expected tiers: {n_tiers}\n"
    f"Non-empty tiers: {np.sum(tier_mask)}\n"
)
        
clicked_x = collect_tier_boundary_clicks(fig_tiers, axs[1])

plt.show(block=False)        
        
input(
    "Close the tier-boundary review window, then press Enter here "
    "(no clicks = accept suggested boundaries).\n"
)

plt.close(fig_tiers)

if len(clicked_x) > 0:
    ex = np.array(clicked_x).astype(int)
    ex[ex<0] = 0
    ex[ex>len(shifted_mean_intensity_profile_z)] = len(shifted_mean_intensity_profile_z)
    tier_detection_method = "manual_override"
    print("new vertical peaks", ex)
else:
    ex = suggested_tier_boundaries
    tier_detection_method = "automatic_peaks"
    print("using ", ex)

print()
print("Tier boundary selection summary")
print(f"Detected left edge:      {left_edge}")
print(f"Detected right edge:     {right_edge}")
print(f"Detected internal:       {suggested_internal}")
print(f"Suggested boundaries:    {suggested_tier_boundaries}")
print(f"Selection method:        {tier_detection_method}")
print(f"Final boundaries used:   {ex}")
print()

tier_segmentation = {}

# ============================================================
# Inspect tier order
# ============================================================

ranges = [
    [ex[i], ex[i + 1]]
    for i in range(len(ex) - 1)
]

tier_images = [
    vol[start:end, :, :].mean(0)
    for start, end in ranges
]

if len(tier_images) == 1:
    preview_images = [tier_images[0]]
else:
    preview_images = [tier_images[0], tier_images[-1]]

image_h, image_w = preview_images[0].shape
aspect = image_w / image_h

image_width_inches = 5
title_space_inches = 2

fig_width = image_width_inches
fig_height = (
    len(preview_images)
    * image_width_inches
    / aspect
) + title_space_inches


fig_tier_order, axs = plt.subplots(
    len(preview_images),
    1,
    figsize=(fig_width, fig_height)
)

if len(preview_images) == 1:
    axs = [axs]

    axs[0].imshow(preview_images[0], cmap="gray")
    axs[0].set_title(
        "Detected one tier only \n"
        f"Slices {ranges[0][0]}-{ranges[0][1]}"
    )
    axs[0].axis("off")

else:

    axs[0].imshow(preview_images[0], cmap="gray")
    axs[0].set_title(
        f"Detected tier: 1 of {len(tier_images)}\n"
        f"Slices {ranges[0][0]}-{ranges[0][1]}"
    )
    axs[0].axis("off")

    axs[1].imshow(preview_images[1], cmap="gray")
    axs[1].set_title(
        f"Detected tier: {len(tier_images)} of {len(tier_images)}\n"
        f"Slices {ranges[-1][0]}-{ranges[-1][1]}"
    )
    axs[1].axis("off")

fig_tier_order.suptitle(
    f"{dataset_folder_name} | Detected tier order preview"
)

tier_order_png = os.path.join(
    diagnostic_figures_path,
    f"{dataset_folder_name}_tier_order_review.png"
)

fig_tier_order.savefig(
    tier_order_png,
    dpi=300,
    bbox_inches="tight"
)

fig_tier_order.tight_layout()

print()
print("A detected-tier preview window is opening.")
print("Use this preview to compare detected tier order against the layout CSV.")
print("After reviewing the preview, close the preview window.")
print("Then answer the tier-order question in the terminal.")
print()

plt.show(block=False)

input("Press Enter after closing the detected-tier preview window...")

plt.close(fig_tier_order)

reverse_detected_tier_order = ask_yes_no(
    "Do the detected tiers appear inverted relative to the layout CSV?\n"
    "Choose yes if detected tier 1 looks like the bottom tier in the CSV.",
    default="n"
)

tier_segmentation["finalized_tier_boundaries"] = [int(v) for v in ex]
tier_segmentation["n_detected_tiers"] = len(ex) - 1
tier_segmentation["tier_detection_method"] = tier_detection_method
tier_segmentation["reverse_detected_tier_order"] = reverse_detected_tier_order

if reverse_detected_tier_order:
    ranges = ranges[::-1]

# ============================================================
# Remove tiers that contain no specimens according to the layout file.
# ============================================================
active_tiers = np.arange(n_tiers)[tier_mask]
ranges = [ranges[i] for i in active_tiers]

active_tier_ids = tier_ids[active_tiers]

tier_segmentation["active_tier_indices"] = [int(v) for v in active_tiers]
tier_segmentation["active_tier_zranges"] = [
    [int(start), int(end)] for start, end in ranges
]


# ============================================================
# Extract the reduced-volume data for each active tier.
# ============================================================
# Downstream divider and cell segmentation operate in .npz space.

SLICES = [
    vol[start:end, :, :]
    for start, end in ranges
]

# Mean projection of each active tier used for divider detection.
I = [x.mean(0) for x in SLICES]

# ============================================================
# Per-tier geometric normalization via rotation
# ============================================================

# This step applies a small per-tier rotation correction before 
# divider detection. The applied rotation angle is recorded in
# run metadata for replication.

normalized_tier_images = []
tier_rotations = []

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

    tier_rotations.append({
        "tier_id": int(active_tier_ids[i]),
        "rotation_angle": best_angle,
        "angle_min": -5,
        "angle_max": 5,
        "angle_step": 0.25,
        "best_score": float(rotation_scores.max()),
    })
    
# ============================================================
# Per-tier divider detection
# ============================================================

"""
Divider detection operates on the mean projection of each active tier.

A local homogeneity image is computed to emphasize specimen and divider
boundaries while suppressing much of the internal texture. The homogeneity
image is thresholded and converted into row- and column-wise occupancy
profiles.

Rows and columns with high occupancy are treated as candidate divider
edges. Neighboring candidate indices are collapsed into edge bands,
paired according to plausible divider widths, and converted into divider
centerlines.

Diagnostic plots are generated to visualize:
1. The homogeneity image.
2. Candidate divider edges.
3. Paired divider edges.
4. Final divider centerlines.

The resulting divider network is used to define specimen extraction
regions for downstream processing.
"""
occupancy_profile_pngs = []
tier_divider_proposal_pngs = []
tier_divider_proposals = []
diagnostic_rows = []
diagnostic_cols = []

for i, normalized_image in enumerate(normalized_tier_images):

    """
    NOTE(dev): window_size=3, percentile_low=2,
    and percentile_high=98 heuristic parameters.

    Smaller windows produce sharper local edge responses.
    Larger windows broaden the homogeneity halos and may merge
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

    homogeneity_image = local_homogeneity_image(
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

    cutoff = np.percentile(homogeneity_image, 20)
    dark_mask = homogeneity_image < cutoff
    row_occupancy, col_occupancy = compute_occupancy_profiles(dark_mask)

    row_starting_threshold, row_min_fraction, row_all_peak_counts, row_n_threshold_retained_peaks, row_prominence_peaks = (estimate_occupancy_threshold_by_peak_distance(row_occupancy))

    col_starting_threshold, col_min_fraction, col_all_peak_counts, col_n_threshold_retained_peaks, col_prominence_peaks = (estimate_occupancy_threshold_by_peak_distance(col_occupancy))

    for peak in row_prominence_peaks:
        diagnostic_rows.append({
            "tier": int(active_tier_ids[i]),
            "row_or_col": "row",
            "peak_index": peak["peak_index"],
            "peak_value": peak["peak_value"],
            "prominence": peak["prominence"],
            "nearest_retained_distance": peak.get(
                "nearest_retained_distance",
                np.nan
            ),
            "stop_reason": peak.get(
                "stop_reason",
                ""
            ),
            "retained_by_prominence": peak["retained_by_prominence"],
            "retained_by_final_threshold": peak["retained_by_final_threshold"],
            "stopping_peak": peak["stopping_peak"],
        })

    for peak in col_prominence_peaks:

        diagnostic_cols.append({
            "tier": int(active_tier_ids[i]),
            "row_or_col": "col",
            "peak_index": peak["peak_index"],
            "peak_value": peak["peak_value"],
            "prominence": peak["prominence"],
            "nearest_retained_distance": peak.get(
                "nearest_retained_distance",
                np.nan
            ),
            "stop_reason": peak.get(
                "stop_reason",
                ""
            ),
            "retained_by_prominence": peak["retained_by_prominence"],
            "retained_by_final_threshold": peak["retained_by_final_threshold"],
            "stopping_peak": peak["stopping_peak"],
        }
    )


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
        min_fraction=row_min_fraction,
        expected_n_pairs=layout_by_tier[int(active_tier_ids[i])]["n_rows"] - 1
    )

    col_centers, col_pairs, col_occupancy, col_candidates, col_candidate_bands, col_band_centers = paired_edge_centerlines(
        dark_mask,
        axis_label="col",
        min_fraction=col_min_fraction,
        expected_n_pairs=layout_by_tier[int(active_tier_ids[i])]["n_cols"] - 1
    )


    # ============================================================
    # Figures and Plots
    # ============================================================

    fig_occupancy, axs = plt.subplots(2, 1, figsize=(20, 12))

    # Row occupancy profile
    axs[0].plot(row_occupancy)

    axs[0].axhline(
        row_min_fraction,
        color="red",
        linestyle="--",
        label=f"threshold = {row_min_fraction:.2f}"
    )

    axs[0].scatter(
        row_candidates,
        row_occupancy[row_candidates],
        color="magenta",
        zorder=5,
        label="candidates"
    )

    for band in row_candidate_bands:
        axs[0].axvspan(
            band[0],
            band[-1],
            alpha=0.2
    )

    axs[0].set_title("Row occupancy profile")
    axs[0].set_xlabel("row index")
    axs[0].set_ylabel("occupancy")
    axs[0].legend()

    # Column occupancy profile
    axs[1].plot(col_occupancy)

    axs[1].axhline(
        col_min_fraction,
        color="red",
        linestyle="--",
        label=f"threshold = {col_min_fraction:.2f}"
    )

    axs[1].scatter(
        col_candidates,
        col_occupancy[col_candidates],
        color="magenta",
        zorder=5,
        label="candidates"
    )

    for band in col_candidate_bands:
        axs[1].axvspan(
            band[0],
            band[-1],
            alpha=0.2
    )

    axs[1].set_title("Column occupancy profile")
    axs[1].set_xlabel("column index")
    axs[1].set_ylabel("occupancy")
    axs[1].legend()

    fig_occupancy.suptitle(
        f"{dataset_folder_name} | Tier {i+1}: occupancy profiles",
        fontsize=14
    )

    occupancy_png = os.path.join(
        diagnostic_figures_path,
        f"{dataset_folder_name}_tier_{active_tier_ids[i]}_occupancy_profiles.png"
    )

    fig_occupancy.savefig(
        occupancy_png,
        dpi=300,
        bbox_inches="tight"
    )

    occupancy_profile_pngs.append(occupancy_png)

    image_h, image_w = normalized_image.shape
    aspect = image_w / image_h

    image_width = 5
    image_height = image_width / aspect

    width_spacing = 2 # 0.5 left margin + 05. right margin + 2*0.5 center spacings between the images
    height_spacing = 3 #(0.5 top margin + 0.5 title + 0.5 space between title an first row of titles + 2*0.5 for the first row of titles + 2*0.25 for the space between the image titles and the images)
    figure_width = (3 * image_width + width_spacing) # Do margins = left and right margins plus the space in between the two images?
    figure_height = (2 * image_height + height_spacing)

    fig_dividers, axs = plt.subplots(2, 3, figsize=(figure_width, figure_height))

    # Normalized image
    axs[0, 0].imshow(normalized_image, cmap="gray")
    axs[0, 0].set_title(f"Normalized image")
    axs[0, 0].axis("off")


    # homogeneity image
    axs[0, 1].imshow(homogeneity_image, cmap="gray")
    axs[0, 1].set_title(f"Tier {i+1}: local homogeneity image")
    axs[0, 1].axis("off")

    # Dark mask
    axs[0, 2].imshow(dark_mask, cmap="gray")
    axs[0, 2].set_title("Dark mask")
    axs[0, 2].axis("off")

    # Candidate divider edges
    axs[1, 0].imshow(normalized_image, cmap="gray")

    for r in row_candidates:
        axs[1, 0].axhline(r, color="magenta", linestyle=":", linewidth=1)

    for c in col_candidates:
        axs[1, 0].axvline(c, color="magenta", linestyle=":", linewidth=1)

    axs[1, 0].set_title("Candidate divider edges")
    axs[1, 0].axis("off")

    # Paired divider edges
    axs[1, 1].imshow(normalized_image, cmap="gray")

    for r1, r2 in row_pairs:
        axs[1, 1].axhline(r1, color="red", linestyle="--")
        axs[1, 1].axhline(r2, color="green", linestyle="--")

    for c1, c2 in col_pairs:
        axs[1, 1].axvline(c1, color="red", linestyle="--")
        axs[1, 1].axvline(c2, color="green", linestyle="--")

    axs[1, 1].set_title("Paired divider edges")
    axs[1, 1].axis("off")

    # Divider centerlines
    axs[1, 2].imshow(normalized_image, cmap="gray")

    for r1, r2 in row_pairs:
        axs[1, 2].axhline(r1, color="red", linestyle="--")
        axs[1, 2].axhline(r2, color="green", linestyle="--")

    for r in row_centers:
        axs[1, 2].axhline(r, color="cyan", linewidth=2)

    for c1, c2 in col_pairs:
        axs[1, 2].axvline(c1, color="red", linestyle="--")
        axs[1, 2].axvline(c2, color="green", linestyle="--")

    for c in col_centers:
        axs[1, 2].axvline(c, color="cyan", linewidth=2)

    axs[1, 2].set_title("Divider centerlines")
    axs[1, 2].axis("off")


    fig_dividers.suptitle(
        f"{dataset_folder_name} | Tier {i+1}: automated divider proposal",
        fontsize=14
    )

    dividers_png = os.path.join(
        diagnostic_figures_path,
        f"{dataset_folder_name}_tier_{active_tier_ids[i]}_tier_divider_proposals.png"
    )

    fig_dividers.savefig(
        dividers_png,
        dpi=300,
        bbox_inches="tight"
    )

    divider_proposal_pngs.append(dividers_png)

    plt.tight_layout()
    plt.show(block=False)

    print()
    print("An automated divider proposal window is open.")
    print("Review the proposed divider locations.")
    print("Close the divider proposal window when you are done reviewing it.")
    print()

    input("Press Enter after closing the automated divider proposal window...")

    plt.close(fig_occupancy)
    plt.close(fig_dividers)

    review_choice = input(
        "\nPress ENTER to accept automated dividers "
        "or type 'm' for manual override: "
    )

    tier_divider_proposals.append({
        "tier_id": int(active_tier_ids[i]),
        "n_expected_rows": layout_by_tier[int(active_tier_ids[i])]["n_rows"],
        "n_expected_cols": layout_by_tier[int(active_tier_ids[i])]["n_cols"],
        "proposed_rows": np.array(row_centers).astype(int).tolist(),
        "proposed_cols": np.array(col_centers).astype(int).tolist(),
        "n_detected_rows": len(proposed_rows) + 1,
        "n_detected_cols": len(proposed_cols) + 1,
        "review_choice": review_choice.strip().lower(),
        "row_starting_threshold": row_starting_threshold,
        "col_starting_threshold": col_starting_threshold,
        "row_final_threshold": row_min_fraction,
        "col_final_threshold": col_min_fraction,
        "row_all_peak_count": row_all_peak_counts,
        "col_all_peak_count": col_all_peak_counts,
        "row_n_prominence_retained_peaks": sum(peak["retained_by_prominence"] for peak in row_prominence_peaks),
        "col_n_prominence_retained_peaks": sum(peak["retained_by_prominence"] for peak in col_prominence_peaks),
        "row_n_threshold_retained_peaks": row_n_threshold_retained_peaks,
        "col_n_threshold_retained_peaks": col_n_threshold_retained_peaks,    
    })

peak_diagnostics_df = pd.DataFrame(
    diagnostic_rows + diagnostic_cols
)

peak_diagnostics_df.to_csv(
    peak_diagnostics_csv,
    index=False
)

# ============================================================
# Divider review / correction layer
# ============================================================
# manual-only mode: proposals are empty
# assisted mode: proposals come from automation

tier_divider_finalized_definitions = []

for i, normalized_image in enumerate(normalized_tier_images):

    tier_id = int(active_tier_ids[i])

    proposed_rows = tier_divider_proposals[i]["proposed_rows"]
    proposed_cols = tier_divider_proposals[i]["proposed_cols"]

    if tier_divider_proposals[i]["review_choice"] == "m":
        final_rows, final_cols = review_dividers(
            image=normalized_image,
            proposed_rows=proposed_rows,
            proposed_cols=proposed_cols,
            title=f"Tier {tier_id}: divider review"
        )

    else:

        final_rows = proposed_rows
        final_cols = proposed_cols
    
    while True:
        expected_rows = layout_by_tier[tier_id]["n_rows"]
        expected_cols = layout_by_tier[tier_id]["n_cols"]

        accepted_rows = len(final_rows) + 1
        accepted_cols = len(final_cols) + 1

        if accepted_rows == expected_rows and accepted_cols == expected_cols:
            break

        print()
        print("The accepted dividers do not match the layout.")
        print(f"Layout expects {expected_rows} rows and {expected_cols} columns.")
        print(f"Current dividers create {accepted_rows} rows and {accepted_cols} columns.")
        print("Please review this tier again.")
        print()

        final_rows, final_cols = review_dividers(
            image=normalized_image,
            proposed_rows=final_rows,
            proposed_cols=final_cols,
            title=f"Tier {tier_id}: divider review"
        )
                
    divider_method = (
        "automatic_accepted"
        if np.array_equal(final_rows, proposed_rows)
        and np.array_equal(final_cols, proposed_cols)
        else "manual_override"
    )    

    print()
    print(f"Tier {tier_id} divider selection summary")
    print(f"Proposed row dividers:   {proposed_rows}")
    print(f"Proposed col dividers:   {proposed_cols}")
    print(f"Selection method:        {divider_method}")
    print(f"Final row dividers used: {final_rows}")
    print(f"Final col dividers used: {final_cols}")
    print()

    tier_divider_finalized_definitions.append({
        "tier_id": tier_id,
        "proposed_row_dividers": proposed_rows,
        "proposed_col_dividers": proposed_cols,
        "final_row_dividers": final_rows.tolist(),
        "final_col_dividers": final_cols.tolist(),
        "divider_method": divider_method,
    })

# ============================================================
# Write extraction plan for surfacing
# ============================================================

extraction_plan_csv = os.path.join(
    output_path,
    f"{dataset_folder_name}_extraction_plan.csv"
)

extraction_rows = []

for i, divider_info in enumerate(tier_divider_finalized_definitions):

    tier_id = divider_info["tier_id"]
    z_start, z_end = ranges[i]

    tier_layout = layout_by_tier[tier_id]["layout"]

    row_dividers = np.array(divider_info["final_row_dividers"]).astype(int)
    col_dividers = np.array(divider_info["final_col_dividers"]).astype(int)
    
    row_dividers = remove_border_dividers(row_dividers, I[i].shape[0], border_margin_fraction=0.03)
    col_dividers = remove_border_dividers(col_dividers, I[i].shape[1], border_margin_fraction=0.03)

    row_edges = np.concatenate(([0], row_dividers, [I[i].shape[0]]))
    col_edges = np.concatenate(([0], col_dividers, [I[i].shape[1]]))
    
    tier_rotation_angle = tier_rotations[i]["rotation_angle"]

    for row_idx in range(tier_layout.shape[0]):
        for col_idx in range(tier_layout.shape[1]):

            specimen_id = tier_layout[row_idx, col_idx]

            if specimen_id == 0:
                continue

            extraction_rows.append({
                "specimen_id": specimen_id,
                "tier_id": tier_id,
                "row_id": row_idx + 1,
                "col_id": col_idx + 1,
                "z_start": int(z_start),
                "z_end": int(z_end),
                "row_start": int(row_edges[row_idx]),
                "row_end": int(row_edges[row_idx + 1]),
                "col_start": int(col_edges[col_idx]),
                "col_end": int(col_edges[col_idx + 1]),
                "tier_rotation_angle": float(tier_rotation_angle),
            })

extraction_plan = pd.DataFrame(extraction_rows)
extraction_plan.to_csv(extraction_plan_csv, index=False)

n_tiers_expected = n_tiers
n_active_tiers = len(active_tier_ids)

n_expected_specimens = int(np.sum([np.sum(layout_by_tier[t]["mask"]) for t in layout_by_tier]))
n_extracted_specimens = len(extraction_rows)
n_extraction_regions = len(extraction_rows)

# ============================================================
# Prepare dataset for surfacing and cleaning
# ============================================================

print()
print("Segmentation is complete and the extraction plan has been created.")
print()
print(
    "The next questions establish the dataset-specific surfacing settings\n"
    "that will be used by 04_surface.py."
)

# ============================================================
# Set ISO value
# ============================================================


print()
print("The first setting needed is the ISO value.")
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

iso = ask(
    "What isovalue would you like to use for this surfacing run?",
    cast=float
)

# ============================================================
# Set padding
# ============================================================

print()
padding = ask(
    "Padding adds a small margin around each extracted specimen box.\n"
    "This helps avoid cutting off specimen edges if the extraction boundaries are slightly tight.\n"
    "The unit is voxels.\n"
    "Press Enter to use the recommended default of 5 voxels.\n"
    "Enter a larger value to include more surrounding material, or a smaller value if you want tighter specimen crops.",
    default=5,
    cast=int
)

# ============================================================
# Ask for cleaning parameters
# ============================================================

print()
print(
    "Now that we have set the surfacing parameters,\n"
    "let's set a few cleaning parameters."
)
print()

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

timer_03_stop = timeit.default_timer()

# ============================================================
# Calculate runtimes
# ============================================================

runtime_03_seconds = timer_03_stop - timer_03_start

print("03_segment.py runtime: ", runtime_03_seconds)

# ============================================================
# Update metadata
# ============================================================

metadata.setdefault("workflow_runtimes", {})
metadata["workflow_runtimes"]["runtime_03_seconds"] = runtime_03_seconds

metadata["03_segment"] = {
    "status": "complete",

    "n_expected_specimens": n_expected_specimens,

    "tier_segmentation": {
        "n_tiers_expected": n_tiers_expected,  
        "n_detected_tiers": tier_segmentation["n_detected_tiers"],
        "left_edge": int(left_edge),
        "right_edge": int(right_edge),
        "suggested_internal": suggested_internal.tolist(),
        "suggested_tier_boundaries": suggested_tier_boundaries.tolist(),
        "finalized_tier_boundaries": tier_segmentation["finalized_tier_boundaries"],
        "tier_detection_method": tier_segmentation["tier_detection_method"],
        "n_active_tiers": n_active_tiers,
        "active_tier_indices": tier_segmentation["active_tier_indices"],
        "active_tier_zranges": tier_segmentation["active_tier_zranges"],
        "reverse_detected_tier_order": tier_segmentation["reverse_detected_tier_order"],
    },

    "tier_normalization": {
        "angle_min": -5,
        "angle_max": 5,
        "angle_step": 0.25,
        "tier_rotations": tier_rotations,
    },

    "divider_review": {
        "tier_divider_proposals": tier_divider_proposals,
        "tier_divider_finalized_definitions": tier_divider_finalized_definitions,
    },

    "extraction_plan": {
        "n_extracted_specimens": n_extracted_specimens,
        "n_extraction_regions": n_extraction_regions,
        "extraction_plan_csv": extraction_plan_csv,
    },

    "diagnostic_outputs": {
        "diagnostic_figures_path": diagnostic_figures_path,

        "diagnostic_figure_filenames": {
            "tier_segmentation_review": os.path.basename(tiers_png),
            "tier_order_review": os.path.basename(tier_order_png),
            "tier_divider_proposals": [
                os.path.basename(f) for f in divider_proposal_pngs
            ],
            "occupancy_profiles": [
                os.path.basename(f) for f in occupancy_profile_pngs
            ],
        },

        "peak_diagnostics_csv": os.path.basename(peak_diagnostics_csv),
    },

    "surfacing_setup": {
        "iso": iso,
        "padding": padding,
    },
    
    "mesh_cleaning_parameters": {
        "dust_cutoff": dust_cutoff,
        "hole_tolerance": hole_tolerance,
    },
}

save_metadata(metadata_path, metadata)

# ============================================================
# Confirm completion
# ============================================================
print()
print("Segmentation complete.")
print("Metadata updated:")
print(metadata_path)
print()

print()
print("Next step options:")
print("1. Continue directly to 04_surface.py for a single dataset.")
print("2. Stop here and manually start 04_surface.py later if you want to")
print("   surface this dataset later or surface multiple datasets in a batch.")
print()

ask_run_next_step("python 04_surface.py", metadata_path)

