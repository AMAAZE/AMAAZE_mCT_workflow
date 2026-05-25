# Workflow Overview

This document describes how the AMAAZE mCT workflow processes CT slice data from raw images to cleaned surface meshes.

The workflow is organized as a sequence of steps. Each step produces outputs that are used by the next step.

---

## Pipeline Summary

The workflow follows this progression:

1. Define orientation and crop region 
2. Build a processed 3D volume 
3. Segment the scan into tiers and specimen regions 
4. Extract specimen subvolumes and generate meshes 
5. Clean meshes for analysis and visualization 

Optional steps:
- Render standardized images of meshes 
- Generate a contact sheet PDF 

---

## Data Flow

The workflow moves from raw slices to final meshes:

```
CT slices (.tif)

        ↓
Orientation + Crop (Step 01)

        ↓
Subsampled volume (.npz) (Step 02)

        ↓
Segmentation plan (.csv) (Step 03)

        ↓
Per-specimen volumes (.npz) + meshes (.ply) (Step 04)

        ↓
Cleaned meshes (Step 05)

        ↓
[Optional] Images + contact sheet (Steps 06–07)
```

Each step depends on outputs from the previous step. Running steps out of order will result in missing file errors.

---

## Step Descriptions

### Step 01 — Orientation and Cropping

**Script:** `01_set_rotation_crop.py` 
**Purpose:** Define how the scan should be interpreted.

- Select a representative slice
- Set orientation (including transpose if needed)
- Apply rotation to align the specimen grid
- Define a crop region around the packaging

This step is interactive and may require multiple runs.

**Output:**
- `controls.txt` (saved in `scanpath`)

---

### Step 02 — Build Volume

**Script:** `02_build_volume.py` 
**Purpose:** Convert the slice stack into a processed 3D volume.

- Applies orientation and crop from Step 01
- Averages slices using `zwindow`
- Resizes each slice to a fixed resolution

**Output:**
- `ct<scan_num>_new.npz`

---

### Step 03 — Segmentation

**Script:** `03_segment.py` 
**Purpose:** Identify tiers and specimen regions.

- Detects vertical tiers in the scan
- Identifies divider structure
- Segments the grid into individual specimen regions
- Uses layout CSV to assign specimen labels

This step includes interactive components:
- Tier boundary confirmation
- Intensity threshold selection

**Output:**
- `CT<scan_num>.csv` (extraction plan)
- `ct<scan_num>_params.txt` (saved parameters)

---

### Step 04 — Surface Extraction

**Script:** `04_surface.py`
**Purpose:** Extract specimen volumes and generate meshes.

- Reconstructs each specimen from the original slice stack
- Applies padding to avoid clipping
- Generates surface meshes using an isosurface threshold (`iso`)

**Outputs:**
- Per-specimen volumes (`.npz`)
- Meshes (`.ply`)
- Mesh overview images (`.png`)
- Saved in: `Meshes/`

---

### Step 05 — Mesh Cleaning

**Script:** `05_clean_meshes.py`
**Purpose:** Improve mesh quality.

- Removes small disconnected components
- Retains the primary specimen geometry

**Output:**
- Cleaned meshes in `Clean_Meshes/`

---

### Step 06 — Render Views *(Optional)*

**Script:** `06_render_views.py`
**Purpose:** Generate standardized images of each specimen.

- Aligns meshes using PCA
- Renders multiple views
- Adds scale bars

**Output:**
- Images saved to `scanphotos/`

---

### Step 07 — Contact Sheet *(Optional)*

**Script:** `07_build_contact_sheet.py`
**Purpose:** Create a visual summary of all specimens.

- Groups rendered images by specimen
- Arranges views into a grid layout
- Exports a multi-page PDF

**Output:**
- `contact_sheet.pdf` (in `scanphotos/`)

---

## Key Concepts

### Orientation Matters

The choices made in Step 01 determine how all downstream steps interpret the scan.
If orientation or cropping is incorrect, segmentation and extraction will also be incorrect.

---

### Layout-Driven Segmentation

The workflow uses a layout CSV to map specimen positions.
Segmentation is not purely automatic—it is guided by expected structure.

---

### Parameter Sensitivity

Some parameters strongly affect results:

- `iso` (surface extraction threshold)
- `zwindow` (slice averaging)
- `voxel_size_mm` (scaling)

These may need adjustment depending on the dataset.

---

### Intermediate Outputs Are Reused

Each step saves outputs to disk.
This allows you to rerun individual steps without restarting the entire workflow.

---

## Typical Usage Pattern

1. Configure a project-specific ` .json`configuration file
2. Run Step 01 and confirm orientation
3. Run Step 02–03 and verify segmentation
4. Adjust parameters if needed
5. Run Step 04–05 to generate final meshes
6. Optionally render images and build a contact sheet

---

## Where Files Are Written

All outputs are written to the directory specified by:

`scanpath`


Within that directory, the workflow creates subfolders such as:

- `Meshes/`
- `Clean_Meshes/`
- `scanphotos/` (optional)

---

## Summary

The workflow is sequential, configuration-driven, and interactive at key steps.
Correct setup in early stages ensures accurate results in later stages.




