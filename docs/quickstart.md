# Quick Start

This guide walks you through setting up your environment and running the AMAAZE mCT workflow.

---

## 1. Navigate to the Project

Open a terminal and move to the project root directory:

```bash
cd /path/to/ProjectRoot
```

---

## 2. Create and Activate a Virtual Environment

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows (Command Prompt or PowerShell)

```bash
python -m venv .venv
.venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## 4. Verify Installation

```bash
python -c "import numpy, pandas, scipy, matplotlib, skimage, cv2, pydicom, joblib; print('core imports ok')"
python -c "from npy_append_array import NpyAppendArray; print('npy_append_array ok')"
python -c "import amaazetools; print('amaazetools ok')"
python -c "import json; json.load(open('user_inputs.json')); print('user_inputs.json ok')"
```

---

## 5. Prepare Your Inputs

Before running the workflow, confirm:

- Your CT slices (`.tif`) are in the folder specified by `slicepath` in `user_inputs.json`
- Your layout `.csv` is correctly referenced in `layoutfile` in `user_inputs.json`
- Your configuration file (`user_inputs.json`) is complete and accurate

---

## 6. Run the Workflow

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

## 7. Daily Use (After Initial Setup)

### macOS / Linux

```bash
cd /path/to/ProjectRoot
source .venv/bin/activate
```

### Windows

```bash
cd C:\path\to\ProjectRoot
.venv\Scripts\activate
```

---

## Notes

- Steps 01 and 03 include interactive components and may require user input.
- If a step fails, review your configuration file and input paths.
- Output files are written to the directory specified by `scanpath`.



