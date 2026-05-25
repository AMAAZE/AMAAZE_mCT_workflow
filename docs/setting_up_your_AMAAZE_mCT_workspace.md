# setting_up_your_AMAAZE_mCT_workspace.md

This document describes a recommended way to set up an `AMAAZE_mCT_workspace` for running and maintaining the AMAAZE microCT surfacing workflow.

This organization makes it easier to manage multiple datasets and projects while keeping workflow code, outputs, and project files organized and separate.

Examples are provided for both Windows and Linux systems. In most cases, macOS users can follow the Linux examples directly.

Creating this workspace requires cloning two GitHub repositories:

- [AMAAZE_mCT_workflow](https://github.com/AMAAZE/AMAAZE_mCT_workflow.git)
- [AMAAZETools](https://github.com/jwcalder/AMAAZETools.git)

---

# Table of Contents

- [Directory Structure](#directory-structure)
- [Why the Workspace is Organized This Way](#why-the-workspace-is-organized-this-way)
- [Steps to Set Up the Workspace](#steps-to-set-up-the-workspace)
  - [Set Up the Directory](#set-up-the-directory)
  - [Verify Git Installation](#verify-git-installation)
  - [Clone the Required Repositories](#clone-the-required-repositories)
  - [Set Up the Virtual Environment](#set-up-the-virtual-environment)
- [Recommended Workflow](#recommended-workflow)
- [Important Considerations](#important-considerations)
  - [Invoking the Correct JSON File](#invoking-the-correct-json-file)
  - [Verifying the AMAAZETools Installation](#verifying-the-amaazetools-installation)
- [Updating and Maintaining the Workspace](#updating-and-maintaining-the-workspace)
  - [Before Updating](#before-updating)
  - [Updating AMAAZE_mCT_workflow](#updating-amaaze_mct_workflow)
  - [Updating AMAAZETools](#updating-amaazetools)
  - [After Updating](#after-updating)
  - [Important Notes](#important-notes)

---

# Directory Structure

AMAAZE_mCT_workspace/
│
├── AMAAZE_mCT_workflow/
│   ├── cloned repository for surfacing workflow
│   ├── setting_up_your_AMAAZE_mCT_workspace.md (this document) lives here
│   ├── requirements.txt lives here
│   └── user_inputs.json (default) lives here
│
├── AMAAZETools/
│   └── cloned dependency repository
│
├── .venv/
│   └── shared Python environment for this workspace
│
└── datasets/
    ├── project_or_dataset_001/
    ├── project_or_dataset_002/
    └── ...

---

# Why the Workspace is Organized This Way

The workspace is designed to separate:

- workflow code
- supporting tools
- project datasets
- and the Python environment

This structure makes the system easier to maintain over time and safer to use across multiple projects, datasets, users, and research groups.

The two cloned GitHub repositories:

- `AMAAZE_mCT_workflow`
- `AMAAZETools`

should remain unchanged whenever possible. Keeping them separate from project datasets allows users to pull future updates from GitHub using `git pull` without risking project files, outputs, or data.

Project datasets should therefore live outside the cloned repositories in the `datasets/` directory. Each project can maintain its own CT slices, configuration files, layouts, outputs, and logs without modifying the workflow repositories themselves.

The shared virtual environment (`.venv`) is also kept at the workspace level so that both repositories use the same Python environment.

---

# Steps to Set Up the Workspace

## Set Up the Directory

Create a root workspace directory that will contain:

- the workflow repository
- the AMAAZETools repository
- the shared virtual environment
- project datasets

Open a terminal in the location where you would like the workspace to live and run:

```bash
mkdir AMAAZE_mCT_workspace
cd AMAAZE_mCT_workspace
```

## Verify Git Installation

Before cloning the repositories, verify that Git is installed:

```bash
git --version
```

If Git is installed, the terminal should return a version number similar to: 

```text
git version 2.53.0
```

If the terminal instead reports that `git` is not recognized or cannot be found, Git must first be installed:

- [Git installation instructions](https://github.com/git-guides/install-git)

## Clone the Required Repositories

The workspace depends on two GitHub repositories:

- AMAAZE_mCT_workflow
- AMAAZETools

From inside `AMAAZE_mCT_workspace/`, clone both repositories by runnning each of the following command lines, pressing Enter after each line:

```bash
git clone https://github.com/AMAAZE/AMAAZE_mCT_workflow.git
git clone https://github.com/jwcalder/AMAAZETools.git
```

Cloning the repositories separately allows the workflow code, supporting tools, and project datasets to remain independently organized, enabling users to easily pull future updates to the workflow code from GitHub without concern over accidental changes to datasets or project files.

## Set Up the Virtual Environment

The virtual environment should exist at the workspace level rather than inside either repository. This allows both repositories to share the same Python environment while remaining independently updatable.

From inside `AMAAZE_mCT_workspace/`, create the virtual environment:

### Windows

```bash
python -m venv .venv
```

### Linux

```bash
python3 -m venv .venv
```

---

# Recommended Workflow

Once you have setup the workspace, this is the recommended workflow.

1. Activate the workspace virtual environment from `AMAAZE_mCT_workspace/`.

2. Navigate into `AMAAZE_mCT_workflow/`.

3. Copy the default `user_inputs.json` into the appropriate project or dataset directory inside `datasets/`.

4. Rename and update the copied JSON file with project-specific paths and settings.

5. Run workflow scripts from inside `AMAAZE_mCT_workflow/`, specifying the appropriate configuration file using `--config`.

For example:

```bash
python 01_set_rotation_crop.py --config ../datasets/project_or_dataset_001/project_or_dataset_001.json
```

This approach keeps project-specific configuration files separate from the workflow repository while allowing multiple datasets and projects to be managed within the same workspace.

---

# Important Considerations

## Invoking the Correct JSON File
The workflow has an example `user_inputs.json` that is the default file used by the workflow. The likelihood is that there will be multiple `.json` files with project or dataset-specific names. Be sure to specify which file is required, for example:

```bash
python 01_set_rotation_crop.py --config ..\datasets\project1\project1.json
```
Note that the '..' at the beginning of the filepath asks the terminal to go up one directory, back to the main workspace and then from there to search within the indicated subdirectories.

## Verifying the `AMAAZETools` Installation

`AMAAZETools` is installed separately from `requirements.txt` using the local cloned repository:

```bash
python -m pip install -e ../AMAAZETools
```

This command assumes the following workspace structure:

```text
AMAAZE_mCT_workspace/
├── AMAAZE_mCT_workflow/
├── AMAAZETools/
└── .venv/
```

and should be run from inside:

```text
AMAAZE_mCT_workspace/AMAAZE_mCT_workflow/
```

To verify that Python can locate `AMAAZETools`, run:

```bash
python -c "import amaazetools; print('amaazetools ok')"
```

---

# Updating and Maintaining the Workspace

The `AMAAZE_mCT_workspace` contains two independently maintained GitHub repositories:

- `AMAAZE_mCT_workflow`
- `AMAAZETools`

Over time, updates to either repository may include:
- bug fixes
- new workflow features
- dependency updates
- changes to scripts or documentation

To retrieve these updates, use `git pull` within each repository.

---

## Before Updating

Before pulling updates:

1. Close any running workflow processes.
2. Ensure the active project dataset has been backed up if necessary.
3. Activate the workspace virtual environment.

From the workspace root:

```bash
cd AMAAZE_mCT_workspace
.venv\Scripts\activate
```

## Updating AMAAZE_mCT_workflow

Navigate into the workflow repository:

```bash
cd AMAAZE_mCT_workflow
```

Pull the latest updates:

```bash
git pull
```

## Updating AMAAZETools

Return to the workspace root:

```bash
cd ..
```

Navigate into the AMAAZETools repository:

```bash
cd AMAAZETools
```

Pull the latest updates:

```bash
git pull
```

## After Updating

After updating either repository, it may be necessary to reinstall dependencies if the update changed package requirements.

From inside AMAAZE_mCT_workflow:

```bash
python -m pip install -r requirements.txt
```

If required, reinstall AMAAZETools into the active virtual environment:

```bash
python -m pip install -e ..\AMAAZETools
```

## Important Notes

Project datasets stored in `datasets/` are intentionally kept outside the repositories and are therefore unaffected by git pull.

Avoid placing active project datasets, outputs, or configuration files directly inside the cloned repositories whenever possible. This reduces the risk of accidental overwrites, merge conflicts, or unintended interactions with future repository updates.
