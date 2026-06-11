# AMAAZE mCT Surfacing Workflow Final Run Report

This report was generated on: **{{ report_timestamp_human }}**

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
| Dataset name: **{{ dataset_folder_name }}** | Specimens identified: **{{ n_specimens_identified }}** |
| Scan number: **{{ scan_num }}** | Specimens extracted: **{{ n_specimens_extracted }}**|
| Number of input slices: **{{ n_slices }}** | Surface meshes generated: **{{ n_meshes_generated }}** |
| Layout file: **{{ layout_filename }}** | Cleaned meshes produced: **{{ n_meshes_cleaned }}** |

### Errors

| Metric | Value |
|---|---|
| Surface-generation errors | **{{ n_surfacing_errors }}** |
| Mesh-cleaning failures | **{{ n_mesh_cleaning_failures }}** |

### Key workflow settings

| Setting | Value |
|---|---|
| In-plane voxel size | **{{ voxel_size_mm }} mm** |
| Voxel (Slice) spacing | **{{ voxel_spacing_mm }} mm** |
| Slice Index Fraction | **{{ slice_index_fraction }}** |
| Rotation angle | **{{ rotation_angle }} degrees** |
| Transpose preview applied | **{{ transpose_preview }}** |
| Crop row range | **{{ rowrng }}** |
| Crop column range | **{{ colrng }}** |
| z-window | **{{ zwindow }}** |
| ISO value | **{{ iso }}** |
| Isovalue selection method | **{{ iso_method }}** |
| Padding | **{{ padding }} voxels** |

### Segmentation summary

| Metric | Value |
|---|---|
| Expected tiers | **{{ n_tiers_expected }}** |
| Active tiers | **{{ n_active_tiers }}** |
| Tier detection method | **{{ tier_detection_method }}** |
| Tier order reversed | **{{ reverse_detected_tier_order }}** |
| Active tier ranges | **{{ active_tier_ranges }}** |
| Expected specimens | **{{ n_specimens_expected }}** |
| Specimens identified | **{{ n_specimens_identified }}** |
| Extraction regions generated | **{{ n_extraction_regions }}** |

### Cleaning parameters
| Parameter | Value |
|---|---|
| Dust cutoff | **{{ dust_cutoff }} vertices** |
| Hole tolerance | **{{ hole_tolerance }}** |


### Surface generation summary

| Setting | Value |
|---|---|
| Specimens extracted | **{{ n_specimens_extracted }}** |
| Meshes generated | **{{ n_meshes_generated }}** |
| Surfacing errors | **{{ n_surfacing_errors }}** |

### Output folder

**{{ output_path }}** 

### CPU allocation

| Workflow stage | CPU cores used | Selection method |
|---|---|---|
| Specimen extraction | **{{ extract_num_cores }}** | {{ extract_num_cores_method }} |
| Surface generation | **{{ surface_num_cores }}** | {{ surface_num_cores_method }} |
| Mesh cleaning | **{{ clean_num_cores }}** | {{ clean_num_cores_method }} |

### Runtime summary

| Workflow step | Interactive / review time | Automated processing time | Total time |
|---|---:|---:|---:|
| 00. Data intake and setup | {{ runtime_00_formatted }} | — | {{ runtime_00_formatted }} |
| 01. Orientation and cropping | {{ runtime_01_formatted }} | — | {{ runtime_01_formatted }} |
| 02. Volume construction | — | {{ runtime_02_formatted }} | {{ runtime_02_formatted }} |
| 03. Segmentation and extraction planning | {{ runtime_03_formatted }} | — | {{ runtime_03_formatted }} |
| 04. Specimen extraction and surfacing | {{ interactive_setup_runtime_04_formatted }} | {{ automated_runtime_04_formatted }} | {{ runtime_04_formatted }} |
| 05. Mesh cleaning | {{ interactive_setup_runtime_05_formatted }} | {{ automated_mesh_cleaning_runtime_05_formatted }} | {{ runtime_05_formatted }} |
| **Workflow total** | **{{ total_interactive_runtime_formatted }}** | **{{ total_automated_runtime_formatted }}** | **{{ total_runtime_formatted }}** |

---

## 00. Data intake and setup

`00_share_data.py`

The workflow began by collecting basic information about the scan dataset and checking that the required input files could be found. This step defines the dataset name, scan number, source folders, layout file, slice stack, voxel scale, and output location used throughout the rest of the workflow. For this run, the dataset was named **{{ dataset_folder_name }}** and processed as scan **{{ scan_num }}**. 

The workflow found **{{ n_slices }}** supported slice files. The first detected slice was **{{ first_slice }}** and the last detected slice was **{{ last_slice }}**. Slice filenames were sorted using their numeric filename indices. The first detected slice index was **{{ first_slice_index }}** and the last detected slice index was **{{ last_slice_index }}**. Slice numbering was evaluated for consecutiveness and found to be **{{ slice_indices_are_consecutive }}**.

The user selected **{{ slice_index_fraction }}** as the representative slice fraction for previewing later workflow steps. The slice fraction defines the relative position of the preview slice within the scan volume. For example, a value of 0.50 selects a slice near the middle of the scan, while smaller values select slices closer to the beginning of the volume.

Voxel scaling was recorded during setup. The in-plane voxel size was **{{ voxel_size_mm }} mm**. The voxel (or slice) spacing was recorded as **{{ voxel_spacing_mm }} mm**. If slice spacing is blank or recorded as `None`, the workflow treated the scan as isotropic and used the in-plane voxel size for both XY and Z scaling. The initial workflow metadata was written to a dataset-specific JSON file and updated throughout the remainder of the workflow. 

### Filepaths

- All output: 
**{{ output_path }}**
- Metadata: 
**{{ metadata_path }}**
- Scan folder: 
**{{ scanpath }}**
- Slice folder: 
**{{ slicepath }}**
- Layout CSV: 
**{{ layoutfile }}**

---

## 01. Orientation and cropping

`01_set_rotation_crop.py`

The purpose of this step was to align the scan with the specimen layout and define the region of the scan that should be retained for processing. Correct orientation and cropping are important because all downstream segmentation, extraction, and surfacing steps depend on these geometric decisions.

The representative slice selected using the slice index fraction (**{{ slice_index_fraction }}**) from the previous step was displayed for user review. The slice index fraction defines the relative position of the preview image within the scan volume.

The user compared the scan orientation against the specimen layout and determined whether the image axes should be transposed. For this run, transpose preview was set to **{{ transpose_preview }}**, indicating whether the image rows and columns were swapped before rotation and cropping.

After orientation review, the scan was rotated to align the package and specimen grid with the image axes. Rotation was performed interactively and visually confirmed by the user. The accepted rotation angle (**{{ rotation_angle }} degrees**) was applied to align the package and specimen grid with the image axes.

Following orientation correction, the user defined a crop region encompassing the area of interest. This crop removed portions of the image that were not relevant to specimen extraction and reduced computational requirements for later processing steps. The accepted crop retained **rows {{ rowrng }}** and **columns {{ colrng }}**, defining the region that was carried forward into all subsequent workflow stages.

The orientation and cropping parameters established in this step were recorded in the workflow metadata and reused during volume construction, segmentation, specimen extraction, and surfacing.

---

## 02. Volume construction

`02_build_subvolume.py`

The purpose of this step was to construct a reduced working volume for segmentation and extraction planning that preserves specimen organization while reducing computational cost. The workflow loaded all slices identified during the data intake step, in this case **{{ n_input_slices }}**, and applied the transpose, orientation and cropping parameters established in Step 01. 

To reduce the size of the working volume, the workflow applied a z-window of **{{ zwindow }}**, which controls how many adjacent slices are combined into a single slice within the reduced volume. A value of 1 preserves every slice, while larger values reduce the number of slices by averaging groups of adjacent slices into a single representative slice. This reduces the size of the working volume along the z-axis. Increasing the z-window improves computational efficiency by reducing memory usage and processing time, but it also reduces axial resolution and may obscure fine structures that change rapidly between neighboring slices. If the total number of slices was not evenly divisible by the selected z-window, the remaining slices were retained and averaged into a final partial window. For this run, the final partial window contained **{{ remainder }} slice(s)**. 

Note: The reduced volume is not intended for final analysis but serves as an efficient representation of the scan for workflow planning and segmentation. 

For this run, the reduced working volume had dimensions **{{ reduced_volume_shape }}**.
 
---
 
## 03. Segmentation and extraction planning

`03_segment.py`

In Step 03, the reduced volume was used to identify package tiers, detect specimen boundaries, and construct the specimen extraction plan. The workflow began by loading the reduced working volume created during Step 02 and the specimen layout CSV (**{{ layout_filename }}**) supplied during data intake. The layout file defines the expected organization of specimens within the package and provides the specimen identifiers used throughout the remainder of the workflow.

### Tier detection

To identify package tiers, the workflow evaluated changes in average density through the scan volume and proposed tier boundaries automatically. These proposed boundaries and the resulting tier order were reviewed by the user before being accepted or user-modified. Tier boundaries define the slice positions that separate adjacent package tiers. The values below are reported as slice indices within the reduced working volume and indicate the locations along the scan (z) axis where one tier ends and the next begins. For example, a boundary at slice 150 indicates that the transition between two adjacent package tiers occurs at the 150th slice of the reduced working volume. For this run, the accepted tier boundaries were:

**{{ tier_boundaries }}**

The accepted tier boundaries define the active tier ranges used during downstream processing. These ranges specify the slice intervals assigned to each tier for divider detection and specimen extraction. The accepted tier boundaries define the active tier ranges used during downstream processing. Each range represents the inclusive span of slices assigned to a single package tier within the reduced working volume and defines the portion of the scan used for divider detection and specimen extraction. For this run, the active tier ranges were:

**{{ active_tier_ranges }}**

### Tier normalization

Each active tier was independently evaluated for small rotational offsets that could interfere with divider detection and then minor rotational corrections were applied where needed to improve alignment. The following per-tier rotation corrections were applied:

**{{ tier_rotation_summary }}**

These adjustments were used only for segmentation and extraction planning and were recorded to ensure that the workflow remains reproducible.

### Divider detection and review

Within each active tier, the workflow identified potential specimen dividers using automated image analysis. Candidate divider locations were proposed automatically and then reviewed by the user, who could accept the suggestions or manually override them. The final accepted divider structure records the row and column divider positions used to partition each active tier into specimen extraction regions. The values below are reported in the coordinate system of the reduced working volume, where row positions correspond to the vertical image axis and column positions correspond to the horizontal image axis within each tier. Together, these divider locations define the boundaries used to separate neighboring specimens during extraction planning.

**{{ divider_summary }}**

### Specimen extraction regions

After tier boundaries and divider locations were finalized, the workflow generated extraction regions corresponding to the specimen positions defined in the layout file and saved the information to an extraction plan. The resulting extraction plan (**{{ extraction_plan_csv }}**) defines the spatial boundaries used during Step 04 to reconstruct specimen-specific subvolumes from the original full-resolution scan data.

---

## 04. Specimen extraction and surfacing

`04_surface.py`

The purpose of this step was to generate an individual research-ready surface mesh (`.ply`) for each extracted specimen. Unlike Step 03, which operated on the reduced working volume, this step returned to the original full-resolution slice stack. The extraction plan generated during segmentation was used to locate each specimen within the scan and reconstruct an independent three-dimensional volume for each of the **{{ n_specimens_extracted }}** specimens identified in the package.

### Voxel geometry

Real-world scale information was applied during reconstruction to ensure that generated meshes retained accurate dimensions. The in-plane voxel size, which defines the physical size of each pixel within a slice, was **{{ voxel_size_mm }} mm**. The slice spacing, which defines the distance between adjacent slices in the scan volume, was **{{ voxel_spacing_mm }} mm**. The voxel geometry for this run was **{{ is_isotropic }}**, indicating whether the in-plane voxel size and slice spacing were equal (isotropic) or different (anisotropic).

### Isovalue selection

Surface generation requires a grayscale threshold, commonly referred to as an isovalue (ISO), to distinguish specimen material from surrounding air and packaging material. The user can input a pre-selected isovalue or use the interactive ISO helper. The isovalue selected for this run (**{{ iso }}**) was applied throughout mesh generation and strongly influences the appearance of the resulting surfaces. Lower values generally include more material and may preserve additional detail, while higher values may reduce noise but remove fine structures. 

### Padding

Padding was applied around each extraction region to reduce the risk of truncating specimen boundaries during reconstruction. The padding value (**{{ padding }} voxels**) was selected by the user and determines how many voxels of additional space are included around each extraction region before surface generation. 

### Specimen extraction

The extraction plan generated during Step 03 was used to reconstruct individual specimen volumes from the original scan. A total of **{{ n_specimens_extracted }}** volumes were extracted.

### Surface generation

After extraction, each specimen volume was converted into a surface mesh using the selected isovalue. During this run, **{{ n_meshes_generated }}** meshes were successfully generated and **{{ n_surfacing_errors }}** were recorded. If any errors occurred, the surfacing error log contains the affected specimen identifiers and diagnostic information that may assist with troubleshooting. The surfacing error log was saved as:

**{{ surfacing_errors_csv }}**

---

## 05. Mesh cleaning

`05_clean_meshes.py`

The purpose of this step was to remove small disconnected mesh fragments produced during surfacing and retain the primary specimen geometry for each reconstructed specimen prior to downstream analysis.

Surface extraction can occasionally generate isolated fragments, small floating components, or other artifacts that are not part of the specimen itself. These artifacts may arise from imaging noise, packaging material, thresholding effects, or incomplete segmentation. The mesh-cleaning stage reduces these artifacts by identifying connected mesh components and retaining the most appropriate specimen geometry.

### Watertightness note

The mesh-cleaning step removes disconnected mesh fragments and retains the primary specimen geometry, but it does not guarantee watertight meshes. This is intentional: making a mesh watertight can require filling holes, closing gaps, or creating new surface geometry that was not directly reconstructed from the scan data.

For scientific transparency, watertight repair should be treated as a separate downstream processing step when needed for specific applications such as 3D printing, volume estimation, finite element analysis, or software that requires closed surfaces.

### Cleaning parameters

Mesh cleaning was performed using dust cutoff (**{{ dust_cutoff }} vertices**) and hole tolerance (**{{ hole_tolerance }}**) parameters. Dust cutoff defines the minimum number of connected vertices a mesh fragment must contain before it is considered a serious candidate for the specimen mesh. When multiple candidate fragments remain, hole tolerance preferentially selects fragments with fewer detected holes. If no candidate meets the hole criterion, the largest remaining fragment is retained.

### Mesh processing results

The workflow evaluated **{{ n_input_meshes }}** meshes generated during Step 04 and applied the mesh-cleaning procedure to each specimen individually. During this run, **{{ n_meshes_cleaned }}** meshes were successfully cleaned and **{{ n_mesh_cleaning_failures }}** mesh-cleaning failures were recorded.

Mesh-cleaning outcomes are summarized below:

**{{ mesh_cleaning_failure_summary }}**

The mesh-cleaning log was saved as:

**{{ mesh_cleaning_log_csv }}**

This log records one row per input mesh, including component counts, hole-detection status, selected component information, success/failure status, and error messages where applicable.

### Cleaned mesh outputs

The cleaned meshes generated during this step represent the final products produced by the AMAAZE mCT surfacing workflow and are intended for downstream visualization, measurement, and scientific analysis. Cleaned meshes were written to:

**{{ clean_mesh_folder }}**

---

## CPU cores and parallelization

Several workflow stages support parallel processing to improve performance. A central processing unit (CPU) is a physical hardware chip mounted on the motherboard of a computer that performs the calculations required by the workflow. Modern CPUs contain multiple processing cores, allowing several tasks to be performed simultaneously. Parallel processing (parallelization) distributes work across multiple CPU cores to reduce overall processing time for computationally intensive workflow steps. 


The workflow provides three opportunities to use parallel processing: specimen extraction, surface generation, and mesh cleaning. During this run, specimen extraction used **{{ extract_num_cores }} cores ({{ extract_num_cores_method }})**, surface generation used **{{ surface_num_cores }} cores ({{ surface_num_cores_method }})**, and mesh cleaning used **{{ clean_num_cores }} cores ({{ clean_num_cores_method }})**.
