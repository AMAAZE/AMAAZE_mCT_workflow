# Step 03: Segment Specimens

This step identifies the internal structure of the scan and determines where each specimen is located.

The goal is to detect the divider grid, segment the scan into regions, and assign each region a label based on the layout `.csv`.

This is one of the most important steps in the workflow.

---

## Script

```bash
python 03_segment.py --config user_inputs.json
```

---

## What This Step Does

`03_segment.py`:

- loads the processed 3D volume from Step 02
- detects vertical tiers in the scan
- identifies divider structure
- segments the scan into individual specimen regions
- assigns specimen labels using the layout `.csv`
- saves a plan for extraction in the next step

This step includes interactive components.

---

## What You Need Before Running It

Before starting, confirm:

- Step 01 produced a correct orientation and crop
- Step 02 completed successfully
- your layout `.csv` matches the physical arrangement of the scan
- divider counts are correct:
  - `n_dividers_row`
  - `n_dividers_col`

If any of these are incorrect, segmentation may still run, but results will be misaligned or mislabeled.

---

## Key Parameters Used in This Step

**`n_dividers_row`** and **`n_dividers_col`**

Define the expected number of dividers in each direction.

These values guide grid detection.

**`row_dividers_override`** and **`col_dividers_override`**

Allow manual specification of divider positions.

Use these only if automatic detection fails.

**`iso_thresholds_override`**

Controls segmentation thresholds.

- null → interactive selection
- [T1, T2, T3] → use provided values

These thresholds define how voxel intensities are grouped during segmentation.  

T1–T2 identify the divider material, while T3 marks the transition into specimen values.  

Together, they help the script distinguish between background, divider structure, and specimen.

**`voxel_probe_n_clicks`**

Number of points used when estimating thresholds interactively.

Default (5) is usually sufficient.

---

## Interactive Steps

### 1. Tier Detection (First Graph)

A graph will appear showing intensity peaks with vertical lines.

You should see:

- peaks corresponding to tiers
- vertical lines intersecting those peaks indicating the tier dividers

If the lines are correct:
→ press Enter

If not:
→ click the tier divider positions from left to right

Your selections override the automatic detection.

### 2. Threshold Selection

A histogram of voxel intensity values will appear. 

This graph is now primarily for visualization, but it can still be useful when choosing `T1`, `T2`, and `T3` values for `iso_thresholds_override` in the `.json`.

Thresholds are typically determined through voxel probing on a representative slice.

You will be prompted to:

- click 5 divider points
- click 5 specimen points

From those clicks, the workflow estimates three thresholds:

- `T1` and `T2` define the divider range
- `T3` marks the lower bound of specimen values

Together, these thresholds help the script distinguish between background, divider structure, and specimen.

Press Enter after each set of clicks when prompted.

### 3. Segmentation Visualization

The script will display images showing:

- divider detection
- segmentation overlays
- bounding boxes for specimen regions

You should see:

- one divider line per expected divider
- boxes aligned with specimen positions
- segmentation matching the physical layout

### 4. Tier-by-Tier Processing

Each tier is processed in sequence.

Close the visualization window to move to the next tier.

---

## What Success Looks Like

A successful segmentation has:

- the correct number of dividers
- clean separation between specimen regions
- bounding boxes aligned with actual specimens
- labels that match the layout `.csv`

---

## Output

This step writes:

- `CT<scan_num>.csv` → extraction plan
- `ct<scan_num>_params.txt` → saved parameters

These files are stored in `scanpath`.

---

## Mental Model

Segmentation is guided, not fully automatic.

- the scan provides structure
- the layout `.csv` provides labels
- your inputs define how the two are aligned

---

## After This Step

Steps 04 and 05 complete the core workflow (volume extraction and mesh cleaning):

- **Step 04 (`04_surface.py`)**
  - Generates per-specimen volumes (`.npz`) and surface meshes (`.ply`)
  - Outputs are written to: `scanpath/Meshes/`

- **Step 05 (`05_clean_meshes.py`)**
  - Cleans meshes and removes small disconnected components
  - Outputs are written to: `scanpath/Clean_Meshes/`

Steps 06 and 07 are optional:

- **Step 06 (`06_render_views.py`)**
  - Generates standardized images of each specimen
  - Outputs are written to: `scanpath/scanphotos/`

- **Step 07 (`07_build_contact_sheet.py`)**
  - Compiles rendered images into a contact sheet PDF
  - Output is written to: `scanpath/scanphotos/` (e.g., `contact_sheet.pdf`)