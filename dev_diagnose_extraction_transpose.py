#!/usr/bin/env python3
"""
dev_diagnose_extraction_transpose.py

Developer diagnostic for AMAAZE mCT workflow.

Purpose
-------
Test whether the hard-coded transpose during specimen extraction contributes
to mirrored meshes.

This script does NOT modify workflow metadata and does NOT write into the
main Meshes or Clean_Meshes folders. It creates a separate diagnostics folder
inside the dataset output folder.

What it does
------------
1. Reads the workflow metadata JSON.
2. Loads the reduced subvolume created by 02_build_volume.py.
3. Reads the extraction plan CSV created by 03_segment.py.
4. Filters the extraction plan to one tier, default tier 2.
5. Extracts each specimen in that tier twice:
   - with_T:     current workflow behavior, crop.T
   - without_T:  diagnostic behavior, crop
6. Saves paired .npy and .ply files for side-by-side comparison.

Example
-------
python dev_diagnose_extraction_transpose.py ^
  --metadata test_data_bonefrags_CT_2/bonefrags_scan2_AMAAZE_outputs/bonefrags_scan2_metadata.json ^
  --tier 2 ^
  --voxel-spacing-mm 0.0046
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from skimage import measure
from skimage.transform import rescale


def normpath(path):
    return os.path.normpath(os.path.expanduser(str(path).strip().strip('"').strip("'")))


def find_first_existing_key(dct, candidates, label):
    for key in candidates:
        if key in dct:
            return key
    raise KeyError(
        f"Could not find a {label} column/key. Tried: {candidates}. "
        f"Available keys: {list(dct.keys())}"
    )


def load_metadata(metadata_path):
    metadata_path = normpath(metadata_path)
    with open(metadata_path, "r") as f:
        return json.load(f)


def resolve_subvolume_file(metadata):
    step02 = metadata.get("02_build_subvolume", {})
    candidates = [
        step02.get("subvolume_file"),
        step02.get("npz_file"),
        step02.get("reduced_subvolume_file"),
    ]

    for candidate in candidates:
        if candidate and os.path.exists(normpath(candidate)):
            return normpath(candidate)

    raise FileNotFoundError(
        "Could not find the reduced subvolume file from metadata['02_build_subvolume']. "
        "Expected a key such as 'subvolume_file'."
    )


def resolve_extraction_plan_csv(metadata):
    step03 = metadata.get("03_segment", {})
    candidate = step03.get("extraction_plan_csv")

    if candidate and os.path.exists(normpath(candidate)):
        return normpath(candidate)

    raise FileNotFoundError(
        "Could not find extraction plan CSV from metadata['03_segment']['extraction_plan_csv']."
    )


def resolve_output_path(metadata, metadata_path):
    step00 = metadata.get("00_share_data", {})
    candidate = step00.get("output_path")

    if candidate:
        return normpath(candidate)

    return os.path.dirname(normpath(metadata_path))


def resolve_voxel_values(metadata, voxel_size_override=None, voxel_spacing_override=None):
    step00 = metadata.get("00_share_data", {})
    step04 = metadata.get("04_surface", {})

    voxel_size_mm = voxel_size_override
    voxel_spacing_mm = voxel_spacing_override

    if voxel_size_mm is None:
        voxel_size_mm = step00.get("voxel_size_mm")

    if voxel_spacing_mm is None:
        voxel_spacing_mm = step00.get("voxel_spacing_mm")

    # Fallbacks, in case a later metadata layout stores them elsewhere.
    surfacing_parameters = step04.get("surfacing_parameters", {})
    if voxel_size_mm is None:
        voxel_size_mm = surfacing_parameters.get("voxel_size_mm")
    if voxel_spacing_mm is None:
        voxel_spacing_mm = surfacing_parameters.get("voxel_spacing_mm")

    if voxel_size_mm is None:
        raise ValueError("voxel_size_mm was not found in metadata. Pass --voxel-size-mm.")

    if voxel_spacing_mm is None:
        print("voxel_spacing_mm not found; using voxel_size_mm as isotropic fallback.")
        voxel_spacing_mm = voxel_size_mm

    return float(voxel_size_mm), float(voxel_spacing_mm)


def resolve_iso(metadata, iso_override=None):
    if iso_override is not None:
        return float(iso_override)

    step04 = metadata.get("04_surface", {})
    surfacing_parameters = step04.get("surfacing_parameters", {})

    iso = surfacing_parameters.get("iso")
    if iso is None:
        iso = metadata.get("00_share_data", {}).get("iso")

    if iso is None:
        raise ValueError("ISO was not found in metadata. Pass --iso.")

    return float(iso)


def load_reduced_volume(npz_path):
    data = np.load(npz_path)

    # Current workflow uses "vol"; some legacy files used "I".
    for key in ("vol", "I"):
        if key in data:
            return data[key]

    # Fallback: if there is exactly one 3D array, use it.
    keys = list(data.keys())
    array_keys = [
        key for key in keys
        if isinstance(data[key], np.ndarray) and data[key].ndim == 3
    ]

    if len(array_keys) == 1:
        return data[array_keys[0]]

    raise KeyError(
        f"Could not identify 3D volume array in {npz_path}. "
        f"Keys found: {keys}"
    )

def write_ascii_ply(path, verts, faces):
    """
    Small self-contained PLY writer so this diagnostic does not depend on
    AMAAZETools mesh writing internals.
    """
    path = normpath(path)

    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        for v in verts:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")

        for tri in faces:
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")


def surface_volume_to_ply(volume, ply_path, iso, dx, dz):
    """
    Surface a temporary specimen volume.

    The workflow convention being tested is:
    volume axes are [z, row_or_x, col_or_y], depending on whether .T was used.

    To respect anisotropic voxels, rescale the z axis by dz/dx and then multiply
    all marching-cubes vertices by dx. This mirrors the legacy surfacing logic.
    """
    volume = np.asarray(volume)

    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {volume.shape}")

    if volume.max() <= iso or volume.min() >= iso:
        raise ValueError(
            f"ISO {iso} is outside the volume range "
            f"[{volume.min()}, {volume.max()}]."
        )

    z_scale = dz / dx

    if z_scale <= 0:
        raise ValueError(f"Invalid z scale dz/dx = {z_scale}")

    # preserve_range-like behavior: rescale only geometry, not intensity.
    working = rescale(
        volume.astype(float),
        (z_scale, 1, 1),
        mode="constant",
        preserve_range=True,
        anti_aliasing=False
    )

    verts, faces, normals, values = measure.marching_cubes(working, level=iso)

    # Convert voxel units into millimeters.
    verts_mm = verts * dx

    # Match the workflow's normal-flip intent by reversing face winding.
    faces = faces[:, ::-1]

    write_ascii_ply(ply_path, verts_mm, faces)


def get_tier_rows(plan, tier_number):
    tier_col = find_first_existing_key(
        plan,
        ["tier", "tier_index", "tier_number", "active_tier", "tier_id"],
        "tier"
    )

    tier_values = plan[tier_col]

    # Try numeric comparison first, then string comparison.
    numeric = pd.to_numeric(tier_values, errors="coerce")
    if numeric.notna().any():
        out = plan[numeric == int(tier_number)].copy()
    else:
        out = plan[tier_values.astype(str) == str(tier_number)].copy()

    if len(out) == 0:
        raise RuntimeError(
            f"No rows found for tier {tier_number} using column '{tier_col}'. "
            f"Available tier values: {sorted(plan[tier_col].astype(str).unique())}"
        )

    return out, tier_col


def extract_volume_for_specimen(I, row, use_transpose):
    row_start_col = find_first_existing_key(
        row,
        ["row_start_padded", "row_start", "row0", "r0"],
        "row start"
    )
    row_end_col = find_first_existing_key(
        row,
        ["row_end_padded", "row_end", "row1", "r1"],
        "row end"
    )
    col_start_col = find_first_existing_key(
        row,
        ["col_start_padded", "col_start", "col0", "c0"],
        "column start"
    )
    col_end_col = find_first_existing_key(
        row,
        ["col_end_padded", "col_end", "col1", "c1"],
        "column end"
    )

    z_start_col = find_first_existing_key(
        row,
        ["tier_z_start", "z_start", "slice_start", "slice_index_start"],
        "z start"
    )
    z_end_col = find_first_existing_key(
        row,
        ["tier_z_end", "z_end", "slice_end", "slice_index_end"],
        "z end"
    )

    r0 = int(row[row_start_col])
    r1 = int(row[row_end_col])
    c0 = int(row[col_start_col])
    c1 = int(row[col_end_col])
    z0 = int(row[z_start_col])
    z1 = int(row[z_end_col])

    if z0 < 0 or z1 > I.shape[0] or z0 >= z1:
        raise ValueError(f"Invalid z range [{z0}, {z1}) for volume shape {I.shape}")

    slices = []

    for z in range(z0, z1):
        crop = I[z, r0:r1, c0:c1]

        if use_transpose:
            crop = crop.T

        slices.append(crop)

    return np.stack(slices, axis=0)


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose whether extraction .T causes mirrored meshes."
    )
    parser.add_argument("--metadata", required=True, help="Path to workflow metadata JSON.")
    parser.add_argument("--tier", type=int, default=2, help="Tier number to diagnose. Default: 2.")
    parser.add_argument("--iso", type=float, default=None, help="Override ISO value.")
    parser.add_argument("--voxel-size-mm", type=float, default=None, help="Override x/y voxel size in mm.")
    parser.add_argument("--voxel-spacing-mm", type=float, default=None, help="Override z slice spacing in mm.")
    parser.add_argument(
        "--output-name",
        default="diagnostics_extraction_transpose",
        help="Diagnostics folder name inside the AMAAZE output folder."
    )

    args = parser.parse_args()

    metadata_path = normpath(args.metadata)
    metadata = load_metadata(metadata_path)

    output_path = resolve_output_path(metadata, metadata_path)
    subvolume_file = resolve_subvolume_file(metadata)
    extraction_plan_csv = resolve_extraction_plan_csv(metadata)
    dx, dz = resolve_voxel_values(
        metadata,
        voxel_size_override=args.voxel_size_mm,
        voxel_spacing_override=args.voxel_spacing_mm
    )
    iso = resolve_iso(metadata, iso_override=args.iso)

    diagnostic_root = os.path.join(output_path, args.output_name, f"tier_{args.tier}")
    with_t_dir = os.path.join(diagnostic_root, "with_T_current_behavior")
    without_t_dir = os.path.join(diagnostic_root, "without_T_diagnostic")

    os.makedirs(with_t_dir, exist_ok=True)
    os.makedirs(without_t_dir, exist_ok=True)

    print()
    print("Diagnostic extraction-transpose test")
    print("------------------------------------")
    print(f"Metadata:        {metadata_path}")
    print(f"Subvolume:       {subvolume_file}")
    print(f"Extraction plan: {extraction_plan_csv}")
    print(f"Tier:            {args.tier}")
    print(f"ISO:             {iso}")
    print(f"dx:              {dx} mm")
    print(f"dz:              {dz} mm")
    print(f"Output folder:   {diagnostic_root}")
    print()

    I = load_reduced_volume(subvolume_file)
    plan = pd.read_csv(extraction_plan_csv)

    tier_rows, tier_col = get_tier_rows(plan, args.tier)

    specimen_col = find_first_existing_key(
        tier_rows,
        ["specimen_id", "specimen", "specimen_name", "label", "id"],
        "specimen id"
    )

    # One row per specimen is expected, but drop duplicates just in case.
    tier_rows = tier_rows.drop_duplicates(subset=[specimen_col]).copy()

    print(f"Using tier column: {tier_col}")
    print(f"Found {len(tier_rows)} specimen(s) in tier {args.tier}.")
    print()

    records = []

    for idx, row in tier_rows.iterrows():
        specimen_id = str(row[specimen_col])

        print(f"Processing {specimen_id}...")

        for use_transpose, outdir, label in [
            (True, with_t_dir, "with_T"),
            (False, without_t_dir, "without_T"),
        ]:
            volume = extract_volume_for_specimen(I, row, use_transpose=use_transpose)

            npy_path = os.path.join(outdir, f"{specimen_id}_{label}.npy")
            ply_path = os.path.join(outdir, f"{specimen_id}_{label}_iso{int(iso)}.ply")

            np.save(npy_path, volume)

            try:
                surface_volume_to_ply(volume, ply_path, iso=iso, dx=dx, dz=dz)
                status = "success"
                error = ""
                print(f"  {label}: saved .npy and .ply")
            except Exception as e:
                status = "surfacing_failed"
                error = str(e)
                print(f"  {label}: saved .npy, but surfacing failed: {e}")

            records.append({
                "specimen_id": specimen_id,
                "tier": args.tier,
                "condition": label,
                "used_crop_transpose": use_transpose,
                "npy_path": npy_path,
                "ply_path": ply_path if status == "success" else "",
                "volume_shape": str(tuple(volume.shape)),
                "volume_min": float(np.min(volume)),
                "volume_max": float(np.max(volume)),
                "iso": iso,
                "voxel_size_mm_dx": dx,
                "voxel_spacing_mm_dz": dz,
                "status": status,
                "error": error,
            })

    summary_csv = os.path.join(diagnostic_root, "diagnostic_summary.csv")
    pd.DataFrame(records).to_csv(summary_csv, index=False)

    print()
    print("Diagnostic complete.")
    print(f"Summary CSV written to: {summary_csv}")
    print()
    print("Open the paired PLY files from:")
    print(f"  {with_t_dir}")
    print(f"  {without_t_dir}")
    print()
    print("Interpretation:")
    print("- If without_T fixes the mirroring, the extraction transpose is the likely culprit.")
    print("- If both are mirrored, look next at preview transpose, rotation/crop mapping, or slice order.")
    print("- If without_T only rotates/swaps axes but does not fix mirroring, the .T is probably only part of the orientation convention.")
    print()


if __name__ == "__main__":
    main()
