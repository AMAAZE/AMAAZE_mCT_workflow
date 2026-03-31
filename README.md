# AMAAZE mCT Workflow

## Overview

This workflow processes micro-CT (mCT) slice stacks into segmented specimens and surface meshes.

The pipeline is designed to:
- define orientation and cropping
- build a 3D volume from slices
- segment specimens using a known layout
- extract specimen subvolumes
- generate and clean surface meshes

Optional steps allow you to render standardized images and create contact sheets for visualization.

---

## Quick Start

Full setup and run instructions are in:

`docs/quickstart.md`

Basic workflow execution:

```bash
python 01_set_rotation_crop.py --config user_inputs.json
python 02_build_volume.py --config user_inputs.json
python 03_segment.py --config user_inputs.json
python 04_surface.py --config user_inputs.json
python 05_clean_meshes.py --config user_inputs.json
```

Optional steps:

```bash
python 06_render_views.py --config user_inputs.json
python 07_build_contact_sheet.py --config user_inputs.json
```

---

## Workflow Structure

The workflow is organized as a sequence of steps:

| Step | Script | Description |
|------|--------|-------------|
| 01 | 01_set_rotation_crop.py | Define orientation and crop region (interactive) |
| 02 | 02_build_volume.py | Build subsampled 3D volume |
| 03 | 03_segment.py | Segment tiers and specimen regions (interactive) |
| 04 | 04_surface.py | Extract subvolumes and generate meshes |
| 05 | 05_clean_meshes.py | Clean meshes and remove artifacts |
| 06 | 06_render_views.py | (optional) Render standardized mesh images |
| 07 | 07_build_contact_sheet.py | (optional) Create PDF contact sheets |

---

## Documentation

Detailed documentation is available in the docs/ directory:

- `docs/quickstart.md` → environment setup and running the workflow
- `docs/workflow_overview.md` → how the full pipeline fits together
- `docs/json_configuration.md` → configuration file reference
- Step-by-step guides for each script (01–07)

---

## What You Need

Before running the workflow, you should have:

- A folder of CT slices (`.tif`)
- A layout CSV describing specimen positions
- A configuration file (`user_inputs.json`)
- A voxel size (either provided directly or via calibration file)

---

## Outputs

The workflow produces:

- Processed 3D volume (`.npz`)
- Segmentation plan (`.csv`)
- Per-specimen volumes (`.npz`)
- Surface meshes (`.ply`)
- Cleaned meshes (`Clean_Meshes/`)

Optional outputs:

- Rendered images (`scanphotos/`)
- Contact sheet (`.pdf`)

---

## Original Source Workflow

This workflow is based on earlier processing code developed by Riley C. W. O'Neill.

Original repository:  
https://github.com/oneil571/AMAAZE-MCT-Processing

---

## Project Location

This version of the workflow is maintained under the AMAAZE organization:  
https://github.com/AMAAZE

---

## Citation

If you use this workflow in your research, please cite:

```bibtex
@article{o2024masse,
  title={En masse scanning and automated surfacing of small objects using Micro-CT},
  author={O'Neill, Riley CW and Yezzi-Woodley, Katrina and Calder, Jeff and Olver, Peter J},
  journal={arXiv preprint arXiv:2410.07385},
  year={2024}
}
```

---

## Credits

This workflow builds on the original mCT processing pipeline developed by Riley C. W. O'Neill.

Refactoring, workflow design, and documentation: 
- Katrina E. Yezzi-Woodley

Additional contributions:  
- Jeff Calder
- Peter J. Olver