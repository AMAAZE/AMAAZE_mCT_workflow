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

metadata_paths = get_metadata_paths_from_command_line_or_user(
    step_name="03_segment",
    allow_batch=False
)

metadata_path = metadata_paths[0]
metadata = load_metadata_if_available(metadata_path)

md = unpack_metadata(metadata)

# ============================================================
# Load subvolume (npz) from 02
# ============================================================

if not os.path.exists(md.subvolume_file):
    raise RuntimeError(
        f"Processed volume not found: {md.subvolume_file}. "
        "Run 02_build_subvolume.py first."
    )

npzdata = np.load(md.subvolume_file)

# Extract data for this workflow from npz
vol = npzdata["vol"]

# ============================================================
# Load scan layout CSV and build scan structure
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
layout = pd.read_csv(md.layoutfile).fillna(0).to_numpy()

# The layout CSV describes only the current dataset.
# Column 0 is the tier number.
if len(layout) == 0:
    raise ValueError(
        "The layout CSV appears to be empty."
        "Check that the CSV describes the current dataset and follows the required format."
    )

scan_structure, tier_ids, tier_mask = build_scan_structure_from_csv(
    layout,
)

n_tiers = len(tier_ids)

# ============================================================
# Prepare diagnostic figure output folder
# ============================================================

diagnostic_figures_path = os.path.join(md.output_path, "03_diagnostic_figures")
os.makedirs(diagnostic_figures_path, exist_ok=True)

peak_diagnostics_csv = os.path.join(
    md.output_path,
    f"{md.dataset_folder_name}_peak_diagnostics.csv"
)

tier_divider_proposals_csv = os.path.join(
    md.output_path,
    f"{md.dataset_folder_name}_tier_divider_proposals.csv"
)

tier_divider_finalized_definitions_csv = os.path.join(
    md.output_path,
    f"{md.dataset_folder_name}_tier_divider_finalized_definitions.csv"
)

extraction_plan_csv = os.path.join(
    md.output_path,
    f"{md.dataset_folder_name}_extraction_plan.csv"
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

redo_tier_division = True

while redo_tier_division:

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

    suggested_tier_boundaries = tier_boundary_result["suggested_boundaries"]
    left_edge = tier_boundary_result["left_edge"]
    right_edge = tier_boundary_result["right_edge"]
    suggested_internal = tier_boundary_result["suggested_internal"]

    z_reduced = np.arange(len(mean_intensity_profile_z))

    ex, tier_detection_method = collect_tier_boundary_clicks(
        dataset_folder_name=md.dataset_folder_name,
        z_reduced=z_reduced,
        shifted_mean_intensity_profile_z=shifted_mean_intensity_profile_z,
        candidate_peaks=vert_pks,
        detected_left_edge=left_edge,
        detected_right_edge=right_edge,
        suggested_boundaries=suggested_tier_boundaries,
        zwindow=md.zwindow,
        n_slices=md.n_slices,
        output_path=diagnostic_figures_path,
        expected_n_boundaries=len(suggested_tier_boundaries),
        max_boundary_value=len(shifted_mean_intensity_profile_z) - 1,
    )

    tier_segmentation = {}

    # ============================================================
    # Inspect tier order
    # ============================================================

    fig_tier_order, axs_tier_order, ranges, tier_images = create_tier_order_preview_figure(
        vol=vol,
        finalized_tier_boundaries=ex,
        dataset_folder_name=md.dataset_folder_name,
        diagnostic_figures_path=diagnostic_figures_path,
    )

    reverse_detected_tier_order = review_detected_tier_order(
        fig=fig_tier_order,
        axs=axs_tier_order,
        tier_images=tier_images,
        ranges=ranges,
    )
    
    plt.close(fig_tier_order)

    if reverse_detected_tier_order is None:
        continue

    redo_tier_division = False

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

tier_subvolumes = [
    vol[start:end, :, :]
    for start, end in ranges
]

# Mean projection of each active tier used for divider detection.
tier_mean_projections = [x.mean(0) for x in tier_subvolumes]

# ============================================================
# Per-tier geometric normalization via rotation
# ============================================================

# This step applies a small per-tier rotation correction before 
# divider detection. The applied rotation angle is recorded in
# run metadata for replication.

normalized_tier_images, tier_rotations = normalize_tier_images(
    tier_mean_projections,
    active_tier_ids,
)

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
    tier_id = int(active_tier_ids[i])

    tier_result = review_tier_dividers(
        normalized_image=normalized_image,
        expected_layout=scan_structure[tier_id],
        dataset_folder_name=md.dataset_folder_name,
        tier_id=tier_id,
        diagnostic_figures_path=diagnostic_figures_path,
    )

        # (NOTE) dev review the next several dictionary lines to make sure they do not belong in the utils somewhere. 
    tier_divider_proposals.append(tier_result)
    occupancy_profile_pngs.append(tier_result["occupancy_profile_png"])
    tier_divider_proposal_pngs.append(tier_result["tier_divider_proposal_png"])
    diagnostic_rows.extend(tier_result["peak_diagnostic_rows"])
    diagnostic_cols.extend(tier_result["peak_diagnostic_cols"])
    
#    fig_dividers, dividers_png = build_tier_divider_proposal_png(
#        normalized_image=normalized_image,
#        homogeneity_image=homogeneity_image,
#        dark_mask=dark_mask,
#        row_candidates=row_candidates,
#        col_candidates=col_candidates,
#        row_pairs=row_pairs,
#        col_pairs=col_pairs,
#        row_centers=row_centers,
#        col_centers=col_centers,
#        dataset_folder_name=md.dataset_folder_name,
#        tier_id=int(active_tier_ids[i]),
#        diagnostic_figures_path=diagnostic_figures_path,
#    )
#    
#    
#    if review_choice == "manual_override":
#    
#        final_rows, final_cols = review_dividers(
#            image=normalized_image,
#            proposed_rows=proposed_rows,
#            proposed_cols=proposed_cols,
#            title=f"Tier {int(active_tier_ids[i])}: manual divider review",
#            dataset_folder_name=md.dataset_folder_name,
#            diagnostic_figures_path=diagnostic_figures_path,
#            tier_id=int(active_tier_ids[i]),
#        )
#
#    else:
#
#        final_rows = proposed_rows
#        final_cols = proposed_cols

peak_diagnostics_df = pd.DataFrame(diagnostic_rows + diagnostic_cols)
peak_diagnostics_df.to_csv(peak_diagnostics_csv, index=False)

tier_divider_proposals_df = pd.DataFrame(tier_divider_proposals)
tier_divider_proposals_df.to_csv(tier_divider_proposals_csv, index=False)

tier_divider_finalized_definitions = []

for tier_result in tier_divider_proposals:
    tier_divider_finalized_definitions.append({
        "tier_id": tier_result["tier_id"],
        "proposed_row_dividers": tier_result["proposed_rows"],
        "proposed_col_dividers": tier_result["proposed_cols"],
        "final_row_dividers": tier_result["final_rows"],
        "final_col_dividers": tier_result["final_cols"],
        "divider_method": (
            "manual_override"
            if tier_result["review_choice"] == "manual_override"
            else "automatic_accepted"
        ),
    })

tier_divider_finalized_definitions_df = pd.DataFrame(tier_divider_finalized_definitions)
tier_divider_finalized_definitions_df.to_csv(tier_divider_finalized_definitions_csv, index=False)

# ============================================================
# Divider review / correction layer
# ============================================================
# Automated proposals and any manual overrides have already been reviewed
# inside the per-tier divider detection loop.
# This section validates and records the finalized divider choices.
#
#tier_divider_finalized_definitions = []  
#
#for i, normalized_image in enumerate(normalized_tier_images):
#
#    tier_id = int(active_tier_ids[i])
#
#    proposed_rows = tier_divider_proposals[i]["proposed_rows"]
#    proposed_cols = tier_divider_proposals[i]["proposed_cols"]
#
#    final_rows = tier_divider_proposals[i]["final_rows"]
#    final_cols = tier_divider_proposals[i]["final_cols"]
#    
#    while True:
#        expected_rows = scan_structure[tier_id]["n_rows"]
#        expected_cols = scan_structure[tier_id]["n_cols"]
#
#        accepted_rows = len(final_rows) + 1
#        accepted_cols = len(final_cols) + 1
#
#        if accepted_rows == expected_rows and accepted_cols == expected_cols:
#            break
#
#        print()
#        print("The accepted dividers do not match the layout.")
#        print(f"Layout expects {expected_rows} rows and {expected_cols} columns.")
#        print(f"Current dividers create {accepted_rows} rows and {accepted_cols} columns.")
#        print("Please review this tier again.")
#        print()
#        
#
#        final_rows, final_cols = review_dividers(
#            image=normalized_image,
#            proposed_rows=final_rows,
#            proposed_cols=final_cols,
#            title=f"Tier {tier_id}: divider review",
#            dataset_folder_name=md.dataset_folder_name,
#            diagnostic_figures_path=diagnostic_figures_path,
#            tier_id=tier_id,
#        )
#                
#    divider_method = (
#        "manual_override"
#        if tier_divider_proposals[i]["review_choice"] == "manual_override"
#        else "automatic_accepted"
#    )    
#
#    print()
#    print(f"Tier {tier_id} divider selection summary")
#    print(f"Proposed row dividers:   {proposed_rows}")
#    print(f"Proposed col dividers:   {proposed_cols}")
#    print(f"Selection method:        {divider_method}")
#    print(f"Final row dividers used: {final_rows}")
#    print(f"Final col dividers used: {final_cols}")
#    print()
#
#    tier_divider_finalized_definitions.append({
#        "tier_id": tier_id,
#        "proposed_row_dividers": proposed_rows,
#        "proposed_col_dividers": proposed_cols,
#        "final_row_dividers": np.array(final_rows).astype(int).tolist(),
#        "final_col_dividers": np.array(final_cols).astype(int).tolist(),        
#        "divider_method": divider_method,
#    })
#
#tier_divider_finalized_definitions_df = pd.DataFrame(tier_divider_finalized_definitions)
#tier_divider_finalized_definitions_df.to_csv(tier_divider_finalized_definitions_csv, index=False)

# ============================================================
# Write extraction plan for surfacing
# ============================================================

extraction_rows = []

for i, divider_info in enumerate(tier_divider_finalized_definitions):

    tier_id = divider_info["tier_id"]
    z_start, z_end = ranges[i]

    tier_layout = scan_structure[tier_id]["layout"]

    row_dividers = np.array(divider_info["final_row_dividers"]).astype(int)
    col_dividers = np.array(divider_info["final_col_dividers"]).astype(int)
    
    row_dividers = remove_border_dividers(row_dividers, tier_mean_projections[i].shape[0], border_margin_fraction=0.03)
    col_dividers = remove_border_dividers(col_dividers, tier_mean_projections[i].shape[1], border_margin_fraction=0.03)

    row_edges = np.concatenate(([0], row_dividers, [tier_mean_projections[i].shape[0]]))
    col_edges = np.concatenate(([0], col_dividers, [tier_mean_projections[i].shape[1]]))
    
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

n_expected_specimens = int(np.sum([np.sum(scan_structure[t]["mask"]) for t in scan_structure]))
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

    "extraction_plan": {
        "n_extracted_specimens": n_extracted_specimens,
        "n_extraction_regions": n_extraction_regions,
        "extraction_plan_csv": extraction_plan_csv,
    },

    "diagnostic_outputs": {
        "diagnostic_figures_path": diagnostic_figures_path,
        "peak_diagnostics_csv": peak_diagnostics_csv,
        "tier_divider_proposals_csv": tier_divider_proposals_csv,
        "tier_divider_finalized_definitions_csv": tier_divider_finalized_definitions_csv,
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

ask_run_next_step("04_surface.py", metadata_path)

