# AMAAZE mCT Surfacing Workflow Final Run Report

This report was generated on: **June 10, 2026 at 10:07 AM**

## Introduction

This report documents the processing history of an AMAAZE mCT surfacing workflow run. Its purpose is to provide an easy-to-read record of the decisions, parameters, outputs, and processing outcomes associated with a particular dataset.

The report begins with a summary followed by descriptions offering a more detailed context for the information found in the summary and in some cases additional metadata. These descriptions are organized according to the major workflow stages. 

This report should be read alongside the accompanying metadata JSON file. The metadata file serves as the primary machine-readable reproducibility record, while this report provides explanatory context intended to help users understand what was done, why it was done, and what products were generated.

If you decide to reproduce this run and wish to retain the outputs from the previous run, please rename the original output folder so that it is not overwritten by subsequent runs.

Unless otherwise noted, all file paths and outputs reported here correspond to the dataset state at the time this report was generated.

---

## Summary sheet

| Original Data and Metadata | Results |
|---|---|
| Dataset name: **synthetic_data** | Specimens identified: **5** |
| Scan number: **1** | Specimens extracted: **5**|
| Number of input slices: **455** | Surface meshes generated: **5** |
| Layout file: **layout_CT_1_synthetic.csv** | Cleaned meshes produced: **5** |

### Errors

| Metric | Value |
|---|---|
| Surface-generation errors | **0** |
| Mesh-cleaning failures | **0** |

### Key workflow settings

| Setting | Value |
|---|---|
| In-plane voxel size | **0.079 mm** |
| Voxel (Slice) spacing | **None mm** |
| Slice Index Fraction | **0.5** |
| Rotation angle | **0.0 degrees** |
| Transpose preview applied | **False** |
| Crop row range | **[32, 316]** |
| Crop column range | **[32, 319]** |
| z-window | **1** |
| ISO value | **3000.0** |
| Isovalue selection method | **manual_entry** |
| Padding | **5 voxels** |

### Segmentation summary

| Metric | Value |
|---|---|
| Expected tiers | **1** |
| Active tiers | **1** |
| Tier detection method | **manual_override** |
| Tier order reversed | **False** |
| Active tier ranges | **[[24, 427]]** |
| Expected specimens | **5** |
| Specimens identified | **5** |
| Extraction regions generated | **5** |

### Cleaning parameters
| Parameter | Value |
|---|---|
| Dust cutoff | **20 vertices** |
| Hole tolerance | **0** |


### Surface generation summary

| Setting | Value |
|---|---|
| Specimens extracted | **5** |
| Meshes generated | **5** |
| Surfacing errors | **0** |

### Output folder

**I:\AISOS_Users\AISOS_AMAAZE_mCT_workspace\AMAAZE_mCT_workflow\test_data_synthetic\synthetic_data_scan1_AMAAZE_outputs** 

### CPU allocation

| Workflow stage | CPU cores used | Selection method |
|---|---|---|
| Specimen extraction | **20** | default |
| Surface generation | **20** | default |
| Mesh cleaning | **6** | default |

### Runtime summary

| Workflow step | Interactive / review time | Automated processing time | Total time |
|---|---:|---:|---:|
| 00. Data intake and setup | 2.5 min | — | 2.5 min |
| 01. Orientation and cropping | 86.7 sec | — | 86.7 sec |
| 02. Volume construction | — | 14.7 sec | 14.7 sec |
| 03. Segmentation and extraction planning | 46.6 sec | — | 46.6 sec |
| 04. Specimen extraction and surfacing | 27.0 sec | 113.5 sec | 2.3 min |
| 05. Mesh cleaning | 23.0 sec | 1.5 sec | 24.6 sec |
| **Workflow total** | **5.6 min** | **2.2 min** | **7.8 min** |

---

## 00. Data intake and setup

`00_share_data.py`

The workflow began by collecting basic information about the scan dataset and checking that the required input files could be found. This step defines the dataset name, scan number, source folders, layout file, slice stack, voxel scale, and output location used throughout the rest of the workflow. For this run, the dataset was named **synthetic_data** and processed as scan **1**. 

The workflow found **455** supported slice files. The first detected slice was **synthetic_ct_0000.tif** and the last detected slice was **synthetic_ct_0454.tif**. Slice filenames were sorted using their numeric filename indices. The first detected slice index was **0** and the last detected slice index was **454**. Slice numbering was evaluated for consecutiveness and found to be **True**.

The user selected **0.5** as the representative slice fraction for previewing later workflow steps. The slice fraction defines the relative position of the preview slice within the scan volume. For example, a value of 0.50 selects a slice near the middle of the scan, while smaller values select slices closer to the beginning of the volume.

Voxel scaling was recorded during setup. The in-plane voxel size was **0.079 mm**. The voxel (or slice) spacing was recorded as **None mm**. If slice spacing is blank or recorded as `None`, the workflow treated the scan as isotropic and used the in-plane voxel size for both XY and Z scaling. The initial workflow metadata was written to a dataset-specific JSON file and updated throughout the remainder of the workflow. 

### Filepaths

- All output: 
**I:\AISOS_Users\AISOS_AMAAZE_mCT_workspace\AMAAZE_mCT_workflow\test_data_synthetic\synthetic_data_scan1_AMAAZE_outputs**
- Metadata: 
**I:\AISOS_Users\AISOS_AMAAZE_mCT_workspace\AMAAZE_mCT_workflow\test_data_synthetic\synthetic_data_scan1_AMAAZE_outputs\synthetic_data_scan1_metadata.json**
- Scan folder: 
**I:\AISOS_Users\AISOS_AMAAZE_mCT_workspace\AMAAZE_mCT_workflow\test_data_synthetic**
- Slice folder: 
**I:\AISOS_Users\AISOS_AMAAZE_mCT_workspace\AMAAZE_mCT_workflow\test_data_synthetic\CT_1_synthetic_slices**
- Layout CSV: 
**I:\AISOS_Users\AISOS_AMAAZE_mCT_workspace\AMAAZE_mCT_workflow\test_data_synthetic\layout_CT_1_synthetic.csv**

---

## 01. Orientation and cropping

`01_set_rotation_crop.py`

The purpose of this step was to align the scan with the specimen layout and define the region of the scan that should be retained for processing. Correct orientation and cropping are important because all downstream segmentation, extraction, and surfacing steps depend on these geometric decisions.

The representative slice selected using the slice index fraction (**0.5**) from the previous step was displayed for user review. The slice index fraction defines the relative position of the preview image within the scan volume.

The user compared the scan orientation against the specimen layout and determined whether the image axes should be transposed. For this run, transpose preview was set to **False**, indicating whether the image rows and columns were swapped before rotation and cropping.

After orientation review, the scan was rotated to align the package and specimen grid with the image axes. Rotation was performed interactively and visually confirmed by the user. The accepted rotation angle (**0.0 degrees**) was applied to align the package and specimen grid with the image axes.

Following orientation correction, the user defined a crop region encompassing the area of interest. This crop removed portions of the image that were not relevant to specimen extraction and reduced computational requirements for later processing steps. The accepted crop retained **rows [32, 316]** and **columns [32, 319]**, defining the region that was carried forward into all subsequent workflow stages.

The orientation and cropping parameters established in this step were recorded in the workflow metadata and reused during volume construction, segmentation, specimen extraction, and surfacing.

---

## 02. Volume construction

`02_build_subvolume.py`

The purpose of this step was to construct a reduced working volume for segmentation and extraction planning that preserves specimen organization while reducing computational cost. The workflow loaded all slices identified during the data intake step, in this case **455**, and applied the transpose, orientation and cropping parameters established in Step 01. 

To reduce the size of the working volume, the workflow applied a z-window of **1**, which controls how many adjacent slices are combined into a single slice within the reduced volume. A value of 1 preserves every slice, while larger values reduce the number of slices by averaging groups of adjacent slices into a single representative slice. This reduces the size of the working volume along the z-axis. Increasing the z-window improves computational efficiency by reducing memory usage and processing time, but it also reduces axial resolution and may obscure fine structures that change rapidly between neighboring slices. If the total number of slices was not evenly divisible by the selected z-window, the remaining slices were retained and averaged into a final partial window. For this run, the final partial window contained **0 slice(s)**. 

Note: The reduced volume is not intended for final analysis but serves as an efficient representation of the scan for workflow planning and segmentation. 

For this run, the reduced working volume had dimensions **[455, 223, 225]**.
 
---
 
## 03. Segmentation and extraction planning

`03_segment.py`

In Step 03, the reduced volume was used to identify package tiers, detect specimen boundaries, and construct the specimen extraction plan. The workflow began by loading the reduced working volume created during Step 02 and the specimen layout CSV (**layout_CT_1_synthetic.csv**) supplied during data intake. The layout file defines the expected organization of specimens within the package and provides the specimen identifiers used throughout the remainder of the workflow.

### Tier detection

To identify package tiers, the workflow evaluated changes in average density through the scan volume and proposed tier boundaries automatically. These proposed boundaries and the resulting tier order were reviewed by the user before being accepted or user-modified. Tier boundaries define the slice positions that separate adjacent package tiers. The values below are reported as slice indices within the reduced working volume and indicate the locations along the scan (z) axis where one tier ends and the next begins. For example, a boundary at slice 150 indicates that the transition between two adjacent package tiers occurs at the 150th slice of the reduced working volume. For this run, the accepted tier boundaries were:

**[24, 427]**

The accepted tier boundaries define the active tier ranges used during downstream processing. These ranges specify the slice intervals assigned to each tier for divider detection and specimen extraction. The accepted tier boundaries define the active tier ranges used during downstream processing. Each range represents the inclusive span of slices assigned to a single package tier within the reduced working volume and defines the portion of the scan used for divider detection and specimen extraction. For this run, the active tier ranges were:

**[[24, 427]]**

### Tier normalization

Each active tier was independently evaluated for small rotational offsets that could interfere with divider detection and then minor rotational corrections were applied where needed to improve alignment. The following per-tier rotation corrections were applied:

**[{'tier_id': 1, 'rotation_angle': 0.0, 'angle_min': -5, 'angle_max': 5, 'angle_step': 0.25, 'best_score': 3.7358053711106063}]**

These adjustments were used only for segmentation and extraction planning and were recorded to ensure that the workflow remains reproducible.

### Divider detection and review

Within each active tier, the workflow identified potential specimen dividers using automated image analysis. Candidate divider locations were proposed automatically and then reviewed by the user, who could accept the suggestions or manually override them. The final accepted divider structure records the row and column divider positions used to partition each active tier into specimen extraction regions. The values below are reported in the coordinate system of the reduced working volume, where row positions correspond to the vertical image axis and column positions correspond to the horizontal image axis within each tier. Together, these divider locations define the boundaries used to separate neighboring specimens during extraction planning.

**[{'tier_id': 1, 'proposed_row_dividers': [76, 151], 'proposed_col_dividers': [76, 151], 'final_row_dividers': [76, 151], 'final_col_dividers': [76, 151], 'divider_method': 'automatic_accepted'}]**

### Specimen extraction regions

After tier boundaries and divider locations were finalized, the workflow generated extraction regions corresponding to the specimen positions defined in the layout file and saved the information to an extraction plan. The resulting extraction plan (**I:\AISOS_Users\AISOS_AMAAZE_mCT_workspace\AMAAZE_mCT_workflow\test_data_synthetic\synthetic_data_scan1_AMAAZE_outputs\synthetic_data_scan1_extraction_plan.csv**) defines the spatial boundaries used during Step 04 to reconstruct specimen-specific subvolumes from the original full-resolution scan data.

---

## 04. Specimen extraction and surfacing

`04_surface.py`

The purpose of this step was to generate an individual research-ready surface mesh (`.ply`) for each extracted specimen. Unlike Step 03, which operated on the reduced working volume, this step returned to the original full-resolution slice stack. The extraction plan generated during segmentation was used to locate each specimen within the scan and reconstruct an independent three-dimensional volume for each of the **5** specimens identified in the package.

### Voxel geometry

Real-world scale information was applied during reconstruction to ensure that generated meshes retained accurate dimensions. The in-plane voxel size, which defines the physical size of each pixel within a slice, was **0.079 mm**. The slice spacing, which defines the distance between adjacent slices in the scan volume, was **None mm**. The voxel geometry for this run was **True**, indicating whether the in-plane voxel size and slice spacing were equal (isotropic) or different (anisotropic).

### Isovalue selection

Surface generation requires a grayscale threshold, commonly referred to as an isovalue (ISO), to distinguish specimen material from surrounding air and packaging material. The user can input a pre-selected isovalue or use the interactive ISO helper. The isovalue selected for this run (**3000.0**) was applied throughout mesh generation and strongly influences the appearance of the resulting surfaces. Lower values generally include more material and may preserve additional detail, while higher values may reduce noise but remove fine structures. 

### Padding

Padding was applied around each extraction region to reduce the risk of truncating specimen boundaries during reconstruction. The padding value (**5 voxels**) was selected by the user and determines how many voxels of additional space are included around each extraction region before surface generation. 

### Specimen extraction

The extraction plan generated during Step 03 was used to reconstruct individual specimen volumes from the original scan. A total of **5** volumes were extracted.

### Surface generation

After extraction, each specimen volume was converted into a surface mesh using the selected isovalue. During this run, **5** meshes were successfully generated and **0** were recorded. If any errors occurred, the surfacing error log contains the affected specimen identifiers and diagnostic information that may assist with troubleshooting. The surfacing error log was saved as:

**I:\AISOS_Users\AISOS_AMAAZE_mCT_workspace\AMAAZE_mCT_workflow\test_data_synthetic\synthetic_data_scan1_AMAAZE_outputs\surfacing_errors.csv**

---

## 05. Mesh cleaning

`05_clean_meshes.py`

The purpose of this step was to remove small disconnected mesh fragments produced during surfacing and retain the primary specimen geometry for each reconstructed specimen prior to downstream analysis.

Surface extraction can occasionally generate isolated fragments, small floating components, or other artifacts that are not part of the specimen itself. These artifacts may arise from imaging noise, packaging material, thresholding effects, or incomplete segmentation. The mesh-cleaning stage reduces these artifacts by identifying connected mesh components and retaining the most appropriate specimen geometry.

### Watertightness note

The mesh-cleaning step removes disconnected mesh fragments and retains the primary specimen geometry, but it does not guarantee watertight meshes. This is intentional: making a mesh watertight can require filling holes, closing gaps, or creating new surface geometry that was not directly reconstructed from the scan data.

For scientific transparency, watertight repair should be treated as a separate downstream processing step when needed for specific applications such as 3D printing, volume estimation, finite element analysis, or software that requires closed surfaces.

### Cleaning parameters

Mesh cleaning was performed using dust cutoff (**20 vertices**) and hole tolerance (**0**) parameters. Dust cutoff defines the minimum number of connected vertices a mesh fragment must contain before it is considered a serious candidate for the specimen mesh. When multiple candidate fragments remain, hole tolerance preferentially selects fragments with fewer detected holes. If no candidate meets the hole criterion, the largest remaining fragment is retained.

### Mesh processing results

The workflow evaluated **5** meshes generated during Step 04 and applied the mesh-cleaning procedure to each specimen individually. During this run, **5** meshes were successfully cleaned and **0** mesh-cleaning failures were recorded.

Mesh-cleaning outcomes are summarized below:

**0 mesh-cleaning failures recorded. See mesh-cleaning log: I:\AISOS_Users\AISOS_AMAAZE_mCT_workspace\AMAAZE_mCT_workflow\test_data_synthetic\synthetic_data_scan1_AMAAZE_outputs\mesh_cleaning_log.csv**

The mesh-cleaning log was saved as:

**I:\AISOS_Users\AISOS_AMAAZE_mCT_workspace\AMAAZE_mCT_workflow\test_data_synthetic\synthetic_data_scan1_AMAAZE_outputs\mesh_cleaning_log.csv**

This log records one row per input mesh, including component counts, hole-detection status, selected component information, success/failure status, and error messages where applicable.

### Cleaned mesh outputs

The cleaned meshes generated during this step represent the final products produced by the AMAAZE mCT surfacing workflow and are intended for downstream visualization, measurement, and scientific analysis. Cleaned meshes were written to:

**I:\AISOS_Users\AISOS_AMAAZE_mCT_workspace\AMAAZE_mCT_workflow\test_data_synthetic\synthetic_data_scan1_AMAAZE_outputs\Clean_Meshes**

---

## CPU cores and parallelization

Several workflow stages support parallel processing to improve performance. A central processing unit (CPU) is a physical hardware chip mounted on the motherboard of a computer that performs the calculations required by the workflow. Modern CPUs contain multiple processing cores, allowing several tasks to be performed simultaneously. Parallel processing (parallelization) distributes work across multiple CPU cores to reduce overall processing time for computationally intensive workflow steps. 


The workflow provides three opportunities to use parallel processing: specimen extraction, surface generation, and mesh cleaning. During this run, specimen extraction used **20 cores (default)**, surface generation used **20 cores (default)**, and mesh cleaning used **6 cores (default)**.
