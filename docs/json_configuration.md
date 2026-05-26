# JSON Configuration

The workflow is controlled by a configuration file (e.g., `user_inputs.json`).

Some values define where your data are located, some define orientation and cropping, and others affect segmentation, mesh generation, and performance.

All user inputs outside of the interactive steps (found in `01_set_rotation_crop.py` and `03_segment.py`) are provided here. 

In normal use, there should be no need to modify the `.py` files.

You can run the workflow using:

```bash
python 01_set_rotation_crop.py --config user_inputs.json
```
You may also create multiple configuration files (e.g., one per scan) and pass them with `--config`.

---

## How to Use This File

Each parameter in the JSON file controls a specific part of the workflow.

### Using `--config`

The `--config` flag tells each script which configuration file to use and where it can be found.

Example:

```bash
python 03_segment.py --config user_inputs_CT_2_tooth.json
```

or 

```bash
python 03_segment.py --config ./scanfolder/user_inputs_CT_2_tooth.json
```

This allows you to:

- keep separate configuration files for different scans
- avoid modifying a single `user_inputs.json` repeatedly

If `--config` is not provided, the script will look for `user_inputs.json` in the project root.

For each parameter below, you’ll find:

- What it controls
- How to choose a value
- Notes (if relevant)

## Example Configuration

```
{
  "scanpath": "./CT_2_bone",
  "slicepath": "./CT_2_bone/AMAAZE_2 Y Slices",
  "scan_num": 2,
  "layoutfile": "./CT_2_bone/layout_CT_2_bone.csv",
  "n_dividers_row": 7,
  "n_dividers_col": 4,
  "slice_index_fraction": 0.7,
  "ang2rot": -206,
  "rowrng": [370, 2600],
  "colrng": [300, 2530],
  "transpose_preview": true,
  "zwindow": 10,
  "iso": 15000,
  "voxel_size_mm": 0.046
}
```

---

## 1. Scan Location and Identity

`scanpath`

What it controls:
Path to the folder for the scan being processed.
Where all outputs are written.

How to choose:
Set this to the folder for the current scan.

`slicepath`

What it controls:
Location of the `.tif` slice stack.

How to choose:
Point to the folder containing your CT slices.

`scan_num`

What it controls:
Used to match entries in the layout CSV and output filenames.

How to choose:
Must match the scan identifier used in your layout file.

`layoutfile`

What it controls:
`.csv` describing specimen layout.

How to choose:
Provide the path to your layout file.

### Required `.csv` layout 

The layout CSV defines how specimens are arranged in the scan.

The file should represent the grid of specimens in the packaging.

Each cell corresponds to one specimen.

### Example Layout

| scan | tier | row | c1 | c2 | c3 | c4 |
|------|------|-----|----|----|----|----|
| 1 | 1 | 1 | Specimen | Specimen | Specimen | Specimen |
| 1 | 1 | 2 | Specimen | Specimen | Specimen | Specimen |
| 1 | 1 | 3 | Specimen | Specimen | Specimen | Empty |
| 1 | 2 | 1 | Specimen | Specimen | Specimen | Specimen |
| 1 | 2 | 2 | Specimen | Specimen | Specimen | Specimen |
| 1 | 2 | 3 | Specimen | Specimen | Specimen | Specimen |

Note: When using multiple tiers it's a good idea to vary a pattern of empty cells across tiers so it is easier to visually inspect later.


### Rules

- The number of grid divisions must match:
  - `n_dividers_row`
  - `n_dividers_col`
- Each entry should be a unique specimen identifier
- The layout must match the physical arrangement in the scan

### How It’s Used

During segmentation:
- The grid structure is detected from the scan
- The layout CSV assigns labels to each segmented region

If the layout does not match the scan:
- specimens may be mislabeled
- extraction regions may be incorrect

---

## 2. Layout and Divider Settings

`n_dividers_row`, `n_dividers_col`

What they control:
Expected number of dividers in each direction.

How to choose:
Count the dividers in your packaging layout.

`row_dividers_override`, `col_dividers_override`

What they control:
Manual divider positions.

How to choose:
Leave as null unless automatic segmentation fails.

Notes:
These override automatic detection in Step 03.

---

## 3. Preview and Orientation (Step 01)

`slice_index_fraction`

What it controls:
Which slice is shown for preview.

How to choose:
Pick a value that shows the packaging clearly (e.g., 0.5–0.7).
The number indicates a percentage location within the `.tif` stack (e.g., 0.5 is half-way through the stack)

`transpose_preview`

What it controls:
Whether the preview image is transposed.

How to choose:
Set to true if the scan appears mirrored relative to the layout in the `.csv`.

`ang2rot`

What it controls:
Rotation angle (degrees).

How to choose:
Adjust until the specimen grid is aligned.
Column dividers should be vertical.
Row dividers should be horizontal.

`rowrng`, `colrng`

What they control:
Crop region.

How to choose:
Define bounds that tightly frame the packaging.
`rowrng` = [top margin, bottom margin] (y-axis)
`colrng` = [left margin, right margin] (x-axis)
(bottom - top) == (right - left)


`allow_slice_gaps`

What it controls:
Whether missing slice indices are allowed.

How to choose:
Set to true if your dataset has gaps.

---

## 4. Segmentation (Step 03)

`iso_thresholds_override`

What it controls:
Manual thresholds for segmentation.

How to choose:
Leave null for interactive selection, or provide values if known.

`voxel_probe_n_clicks`

What it controls:
Number of clicks used to estimate thresholds.

How to choose:
Default (5) is usually sufficient.

---

## 5. Volume Construction (Step 02)

`zwindow`

What it controls:
Number of slices averaged together.

How to choose:

- Larger → smoother, faster, lower resolution
- Smaller → more detail, slower

---

## 6. Surface Extraction (Step 04)

`iso`

What it controls:
Threshold used to generate meshes.

How to choose:

Too low → noisy or incorrect surfaces
Too high → missing geometry

**Intuition:**
- Think of `iso` as the boundary between “material” and “non-material”
- Lower values include more data (risk: noise)
- Higher values include less data (risk: missing structure)

Adjust based on dataset.

`padding`

What it controls:
Extra margin around each specimen.

How to choose:
Small positive value (e.g., 5) prevents clipping.

---

## 7. Scaling

`voxel_size_mm`

What it controls:
Physical size of each voxel within the slice plane, in millimeters. This is the X/Y voxel size.

How to choose:
Use the known in-plane voxel size from the scan metadata or calibration file.

`voxel_spacing_mm`

What it controls:
Physical spacing between slices in the Z direction, in millimeters.

How to choose:
Leave as `null` when the scan has isotropic voxels, meaning Z spacing is the same as X/Y voxel size.

Provide a number when the scan has anisotropic voxel dimensions, such as some medical CT/DICOM-derived datasets where slice spacing differs from in-plane pixel size.

Examples:

```json
"voxel_size_mm": 0.079,
"voxel_spacing_mm": null
```
```json
"voxel_size_mm": 0.079,
"voxel_spacing_mm": 0.300
```

Notes:
This affects mesh proportions during surfacing. Running the values listed in the examples above illustrates this difference.

---

## 8. Performance and CPU Usage

These parameters control how many CPU cores are used in different stages of the workflow.

### What is a CPU core?

A CPU core is a processing unit in your computer.  
Using more cores allows multiple operations to run in parallel, which can speed up processing.

### Why are there separate settings?

Different steps in the workflow have different computational demands:

- **Extraction** (Step 04): reading and slicing image data
- **Surfacing** (Step 04): generating meshes (memory-intensive)
- **Cleaning** (Step 05): processing mesh geometry

These are separated so you can:
- increase speed where safe
- reduce memory usage where needed

### Parameters

#### `extract_ncores`
Used during specimen extraction (Step 04).

- Controls how many specimens are processed at the same time
- Generally safe to increase

#### `surface_ncores`
Used during mesh generation (Step 04).

- Most memory-intensive step
- Increasing this too much can cause crashes

**Guideline:**
- Start low (e.g., 2–4 cores)
- Increase only if stable

#### `clean_ncores`
Used during mesh cleaning (Step 05).

- Typically lightweight
- Can usually use more cores safely

### How to choose values

- `null` → uses the workflow’s default behavior (may use many cores)
- For small datasets, this is usually fine
- For large datasets, especially during surfacing, this may cause memory issues
- Small datasets → higher values are usually fine
- Large datasets → use fewer cores to avoid memory issues

If the workflow crashes during Step 04:
→ reduce `surface_ncores`

### The mental model

- Extraction = CPU-bound → parallelize freely
- Surfacing = memory-bound → be careful
- Cleaning = light → safe

---

## 9. Mesh Cleanup (Step 05)

`dust_cutoff`

What it controls:
Minimum component size kept.

`hole_tolerance`

What it controls:
How aggressively holes are filled.

---

## Key Parameters to Focus On

If you are new to the workflow, pay closest attention to:

- `slicepath`
- `layoutfile`
- `ang2rot`
- `rowrng`, `colrng`
- `transpose_preview`
- `n_dividers_row`, `n_dividers_col`
- `zwindow`
- `iso`
- `voxel_size_mm`
- `voxel_spacing_mm`

These have the largest impact on results.

---

## Notes
- Most users will adjust parameters iteratively.
- Early steps (orientation and segmentation) strongly influence final outputs.
- Incorrect configuration can still produce outputs, but they may be misaligned or incorrect.

