# Quick Start

This document provides a quick-reference setup guide for running the AMAAZE microCT surfacing workflow.

---

## 1. Navigate to the Workspace

Open a terminal and navigate to `AMAAZE_mCT_workspace/`.

### Linux

```bash
cd /path/to/AMAAZE_mCT_workspace
```

### Windows (Command Prompt or PowerShell)

```bash
cd C:\path\to\AMAAZE_mCT_workspace
```

---

## 2. Activate the Virtual Environment

### Linux

```bash
source .venv/bin/activate
```

### Windows (Command Prompt or PowerShell)

```bash
.venv\Scripts\activate
```

---

## 3. Navigate to the Workflow Repository

```bash
cd AMAAZE_mCT_workflow
```
---

## 4. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e ../AMAAZETools
```

---

## 5. Verify Installation

```bash
python -c "import numpy, pandas, scipy, matplotlib, skimage, cv2, pydicom, joblib; print('core imports ok')"
python -c "from npy_append_array import NpyAppendArray; print('npy_append_array ok')"
python -c "import amaazetools; print('amaazetools ok')"
python -c "import json; json.load(open('user_inputs.json')); print('user_inputs.json ok')"
```

---

## 6. Prepare Your Inputs

Before running the workflow, confirm:

- Copy the example `user_inputs.json` into the appropriate project or dataset directory inside `datasets/`
- Rename and update the copied configuration file for the current dataset
- Confirm that `slicepath` and `layoutfile` point to the correct project files

---

## 7. Run the Workflow

```bash
python 01_set_rotation_crop.py --config ../datasets/project_or_dataset_001/project_or_dataset_001.json
python 02_build_volume.py --config ../datasets/project_or_dataset_001/project_or_dataset_001.json
python 03_segment.py --config ../datasets/project_or_dataset_001/project_or_dataset_001.json
python 04_surface.py --config ../datasets/project_or_dataset_001/project_or_dataset_001.json
python 05_clean_meshes.py --config ../datasets/project_or_dataset_001/project_or_dataset_001.json
```

Optional steps:

```bash
python 06_render_views.py --config user_inputs.json
python 07_build_contact_sheet.py --config user_inputs.json
```

---

## 8. Daily Use (After Initial Setup)

### Linux

```bash
cd /path/to/AMAAZE_mCT_workspace
source .venv/bin/activate
cd AMAAZE_mCT_workflow
```

### Windows

```bash
cd C:\path\to\AMAAZE_mCT_workspace
.venv\Scripts\activate
cd AMAAZE_mCT_workflow
```

---

## Notes

- Steps 01 and 03 include interactive components and may require user input.
- If a step fails, review your configuration file and input paths.
- Output files are written to the directory specified by `scanpath`.



