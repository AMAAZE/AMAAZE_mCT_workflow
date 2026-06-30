#!/usr/bin/env python3

"""
03_segment.py

Original processing logic: RileyWilde
Refactoring and workflow design: Katrina E. Yezzi-Woodley

Interactive segmentation step for the AMAAZE mCT surfacing workflow.

This script loads the reduced working volume created by 02_build_subvolume.py
and uses the layout CSV to guide segmentation into tiers and specimen regions.

This step:
- detects and reviews tier boundaries,
- confirms the detected tier order,
- applies per-tier geometric normalization,
- detects and reviews row and column specimen dividers,
- writes diagnostic figures and divider-review CSVs,
- creates the specimen extraction plan,
- records surfacing and mesh-cleaning parameters for 04_surface.py,
- and updates the workflow metadata JSON.
"""

# ============================================================
# Configuration and imports
# ============================================================

from utils import *

# ============================================================
# Load metadata
# ============================================================

timer_03_start = timeit.default_timer()

TOTAL_QUESTIONS_03 = 5

print_terminal_header("Step 4 of 5: Segment Specimen Regions")

print("In this step, you will segment the reduced working volume")
print("into tiers and specimen regions.")
print()
print("You will:")
print()
print("    1. Provide the workflow metadata JSON")
print("    2. Review and confirm tier boundaries")
print("    3. Review detected tier order")
print("    4. Review specimen divider proposals")
print("    5. Set surfacing and mesh-cleaning parameters")
print()
print("Steps 2, 3, and 4 use interactive preview windows.")
print()
print("When you're ready, press Enter to begin.")

input("> ")

print_question_header("Workflow Metadata JSON", 1, TOTAL_QUESTIONS_03)

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

vol = npzdata["vol"]

# ============================================================
# Load scan layout CSV and build scan structure
# ============================================================

layout = pd.read_csv(md.layoutfile).fillna(0).to_numpy()

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

    if reverse_detected_tier_order is None:
        continue

    redo_tier_division = False

    tier_segmentation["finalized_tier_boundaries"] = [int(v) for v in ex]
    tier_segmentation["n_detected_tiers"] = len(ex) - 1
    tier_segmentation["tier_detection_method"] = tier_detection_method
    tier_segmentation["reverse_detected_tier_order"] = reverse_detected_tier_order

    if reverse_detected_tier_order:
        ranges = ranges[::-1]
    
    selected_tier_order_png = os.path.join(
        diagnostic_figures_path,
        f"{md.dataset_folder_name}_selected_tier_order.png"
    )

    fig_tier_order.savefig(
        selected_tier_order_png,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig_tier_order)

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

normalized_tier_images, tier_rotations = normalize_tier_images(
    tier_mean_projections,
    active_tier_ids,
)

# ============================================================
# Per-tier divider detection
# ============================================================

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

    tier_divider_proposals.append(tier_result)
    occupancy_profile_pngs.append(tier_result["occupancy_profile_png"])
    tier_divider_proposal_pngs.append(tier_result["tier_divider_proposal_png"])
    diagnostic_rows.extend(tier_result["peak_diagnostic_rows"])
    diagnostic_cols.extend(tier_result["peak_diagnostic_cols"])
    
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

print_question_header("Surfacing Parameters: ISO Value", 2, TOTAL_QUESTIONS_03)

print()
print(
    "Segmentation is complete and the extraction plan has been created.\n"
    "The next questions establish the dataset-specific surfacing settings\n"
    "that will be used by 04_surface.py."
)
print()

print()
print(
    "The first setting needed is the ISO value.\n"
    "    - An isovalue (ISO) is the grayscale threshold used to \n"
    "      decide which voxels belong to the specimen and which \n"
    "      belong to the surrounding background.\n\n"
    "    - Lower isovalues capture detail, but can also introduce more noise.\n\n"
    "    - Higher isovalues reduce noise but can remove real specimen structure.\n\n"
)

print()
iso = ask(
    "What isovalue would you like to use for this surfacing run?",
    cast=float
)

# ============================================================
# Set padding
# ============================================================
print_question_header("Surfacing Parameters: Padding", 3, TOTAL_QUESTIONS_03)

print()
padding = ask(
    "Padding adds a small margin around each extracted specimen box.\n"
    "This helps avoid cutting off specimen edges if the extraction boundaries are slightly tight.\n\n"
    "Entering a larger value includes more surrounding material, whereas a smaller value produces tighter specimen crops.\n\n"
    "What padding value would you like to use?\n"
    "Press Enter to use the recommended default of 5 voxels.\n",
    default=5,
    cast=int
)

# ============================================================
# Ask for cleaning parameters
# ============================================================

print_question_header("Cleaning Parameters: Dust Cutoff", 4, TOTAL_QUESTIONS_03)

print()
print(
    "Now that we have set the surfacing parameters,\n"
    "let's set a few cleaning parameters.\n"
)
print()

dust_cutoff = ask(
    "Dust cutoff removes small disconnected mesh fragments.\n\n"
    "The default is 20 vertices, meaning fragments with 20 vertices or fewer will be deleted.\n\n" 
    "Enter a dust cutoff value or press Enter to accept the default.\n",
    default=20,
    cast=int
)

print_question_header("Cleaning Parameters: Hole Tolerance", 5, TOTAL_QUESTIONS_03)

hole_tolerance = ask(
    "Hole tolerance helps identify the correct specimen during mesh cleaning.\n\n"
    "It limits how many holes a specimen mesh may have\n"
    "and still be selected during mesh cleaning.\n"
    "Lower values are more strict.\n\n"
    "Enter a hole tolerance value or press Enter to accept the default.",
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

print_step_complete_header("Step 4 Complete")

print("Segmentation complete.")
print()
print(f"Setup took {format_runtime(runtime_03_seconds)} to complete.")
print()
print("Metadata updated:")
print()
print(f"    {metadata_path}")
print()
print("Important: Use this metadata JSON to continue the workflow later.")
print()

ask_run_next_step("04_surface.py", metadata_path)

