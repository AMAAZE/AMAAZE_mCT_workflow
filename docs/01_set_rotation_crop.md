# Step 01: Set Rotation and Crop

This step defines how the workflow should interpret the scan before volume construction and segmentation begin.

The goal is to produce a preview that matches the physical arrangement of the packaging and the specimen layout in the `.csv` file.

The choices made here affect all downstream steps.

---

## Script

```bash
python 01_set_rotation_crop.py --config user_inputs.json
```

---

## What This Step Does

`01_set_rotation_crop.py`:

- loads a representative slice from the `.tif` stack
- applies the current preview settings from the configuration file
- displays the preview image
- saves the selected rotation and crop settings for downstream use

This is an interactive setup step.

You may need to run it multiple times before the preview looks correct.

---

## What You Need Before Running It

Before starting, make sure the following values in `user_inputs.json` are set:

- `slicepath`
- `slice_index_fraction`
- `ang2rot`
- `rowrng`
- `colrng`
- `transpose_preview`

You should also have your layout `.csv` available so you can compare the preview to the intended specimen arrangement.

Recommendation: Start `ang2rot` at 0 and the `rowrng` and `colrng` at [0, 5000] so you can see the full image "as-is".

---

## Key Parameters Used in This Step

**`slice_index_fraction`**

Controls which slice is shown for preview.

Choose a value that shows the packaging clearly.
A value of 0.5 means the script will preview a slice about halfway through the stack.

**`transpose_preview`**

Controls whether the preview image is transposed.

Use this when the preview appears mirrored relative to the specimen layout in the `.csv`.

If the arrangement looks flipped, try switching this between true and false and rerun the step.

**`ang2rot`**

Controls the rotation angle applied to the preview.

Adjust this until:

- column dividers appear vertical
- row dividers appear horizontal
- the grid visually matches the layout

**`rowrng`** and **`colrng`**

These define the crop region.

- rowrng = [top, bottom]
- colrng = [left, right]

The crop should tightly frame the packaging without cutting into specimen regions.

For this workflow, the crop should be square:

(bottom - top) = (right - left)

This matters because the next step resizes the cropped region to a fixed square shape.

This constraint reflects how the workflow currently models the physical packaging as a regular grid. 

A square crop preserves that structure during resizing and segmentation.

Future versions of the workflow may relax this requirement as handling of packaging geometry becomes more flexible.

---

## Recommended Workflow

### 1. Start with a broad preview

If you do not yet know the correct crop, begin with a large region so the full packaging is visible.

Example:

```json
"rowrng": [0, 5000],
"colrng": [0, 5000]
```

Then refine from there.

### 2. Check transpose first

Before worrying about exact rotation, determine whether the preview is oriented correctly relative to the layout `.csv`.

Ask:

- Does the specimen arrangement match the expected pattern?
- Does the left side of the preview correspond to the left side of the layout?
- Do distinctive empty cells or specimen positions line up?

If not, try changing `transpose_preview`.

### 3. Adjust rotation

Once transpose is correct, adjust `ang2rot` until the packaging grid is aligned.

You are aiming for:

- straight-looking row dividers
- straight-looking column dividers
- a preview that visually matches the expected grid

This usually takes a few iterations.

### 4. Tighten the crop

After orientation is correct, refine `rowrng` and `colrng` so the crop:

- includes the full packaging
- excludes unnecessary surrounding background
- remains square

### 5. Rerun until satisfied

This step is often iterative.
It is normal to rerun it several times before moving on.

---

## What Success Looks Like

A good Step 01 result has all of the following:

- the scan is not mirrored relative to the layout
- row dividers appear horizontal
- column dividers appear vertical
- the packaging is tightly framed
- the crop is square
- the layout pattern looks correct by eye

If this step looks wrong, later steps may still run, but segmentation and extraction may be incorrect.

---

## Output

This step writes:

`controls.txt`

to the directory specified by `scanpath`.

This file stores the chosen preview orientation, rotation, and crop settings for later steps.

---

## Common Decision Points

### When should I change `transpose_preview`?

Change it when the preview appears flipped relative to the physical arrangement in the layout `.csv`.

This is usually easier to diagnose using distinctive occupied or empty positions in the layout.

### When should I change `ang2rot`?

Change it when the packaging grid is tilted and the dividers are not aligned with the image axes.

### When should I change `rowrng` and `colrng`?

Change them when:

- too much background is included
- part of the packaging is cut off
- the crop is not square

### What if I am unsure whether the preview is correct?

Do not move on yet.

Step 01 is foundational.
It is better to spend extra time here than to debug mislabeled or misaligned outputs later.

---

## After This Step

Once the preview is correctly oriented and cropped, proceed to:

`02_build_volume.py`

This step:

- applies the selected orientation and crop to the full `.tif` stack
- constructs a processed 3D volume
- prepares the data for segmentation

There is no user interaction in this step.

Once it completes successfully, proceed directly to:

`03_segment.md`


