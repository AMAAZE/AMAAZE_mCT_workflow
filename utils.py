#!/usr/bin/env python3

"""
utils.py

Original processing logic: RileyWilde
Refactoring and workflow design: Katrina E. Yezzi-Woodley

Shared imports, configuration, and helper functions for the AMAAZE mCT workflow.

Developer-facing module. End users should not need to edit this file.
"""

# ============================================================
# IMPORTS
# Shared libraries used throughout the workflow.
# ============================================================

import json
import os
import multiprocessing
import timeit
import re

import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

import cv2 as cv
import skimage.io as io
import pydicom

import datetime
import shutil
import subprocess

import amaazetools.trimesh as tm
from amaazetools import dicom

from joblib import Parallel, delayed
from npy_append_array import NpyAppendArray

from skimage import measure
from skimage.transform import rotate, rescale
from scipy.signal import find_peaks
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix


# ============================================================
# RUN METADATA HELPERS
# Used throughout the workflow.
# ============================================================

def load_metadata_if_available(metadata_path):
    """
    Load workflow metadata if it already exists; otherwise return None.
    """
    if metadata_path is None:
        return None

    metadata_path = os.path.normpath(metadata_path)

    if not os.path.exists(metadata_path):
        return None

    with open(metadata_path, "r") as f:
        return json.load(f)


def save_metadata(metadata_path, metadata):
    """
    Save workflow metadata to disk.
    """
    metadata_path = os.path.normpath(metadata_path)

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
        
def count_csv_rows(csv_path):
    """
    Count data rows in a CSV file. Returns 0 if the file is missing or empty.
    """
    if not os.path.exists(csv_path):
        return 0

    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return 0

    return len(df)


def current_timestamp_for_filename():
    """
    Return a timestamp safe for filenames.
    """
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def render_text_template(template_path, values):
    """
    Fill a simple markdown template using {{ key }} placeholders.
    """
    with open(template_path, "r") as f:
        text = f.read()

    for key, value in values.items():
        text = text.replace("{{ " + key + " }}", str(value))

    return text


def convert_markdown_text_to_pdf(markdown_text, pdf_path):
    """
    Convert rendered markdown text to PDF using pandoc.
    Returns True if the PDF was created, False otherwise.
    """
    if shutil.which("pandoc") is None:
        return False

    try:
        subprocess.run(
            ["pandoc", "-", "-o", pdf_path],
            input=markdown_text,
            text=True,
            check=True
        )
        return True
    except subprocess.CalledProcessError:
        return False

def format_runtime(seconds):
    """
    Convert runtime seconds into a human-readable report string.
    """
    seconds = float(seconds)

    if seconds < 120:
        return f"{seconds:.1f} sec"

    minutes = seconds / 60

    if minutes < 120:
        return f"{minutes:.1f} min"

    hours = minutes / 60
    return f"{hours:.2f} hr"

def write_final_run_report(metadata):
    """
    Fill the final run report template and write a timestamped PDF report
    to the dataset output folder.

    The markdown template stays in the workflow folder.
    """
    step00 = metadata["00_share_data"]
    step01 = metadata["01_set_rotation_crop"]
    step02 = metadata["02_build_subvolume"]
    step03 = metadata["03_segment"]
    step04 = metadata["04_surface"]
    step05 = metadata["05_clean_meshes"]

    output_path = step00["output_path"]
    dataset_name = step00["dataset_name"]
    scan_num = step00["scan_num"]

    timestamp = current_timestamp_for_filename()
    
    report_timestamp_human = datetime.datetime.now().strftime(
    "%B %d, %Y at %I:%M %p"
    )

    template_path = os.path.join(
        os.path.dirname(__file__),
        "final_run_report_template.md"
    )

    report_base = f"final_run_report_{dataset_name}_scan{scan_num}_{timestamp}"
    markdown_path = os.path.join(output_path, report_base + ".md")
    pdf_path = os.path.join(output_path, report_base + ".pdf")

    tier_segmentation = step03["tier_segmentation"]
    tier_normalization = step03["tier_normalization"]
    divider_review = step03["divider_review"]
    surfacing_parameters = step04["surfacing_parameters"]
    surfacing_parallelization = step04["parallelization"]
    cleaning_parameters = step05["mesh_cleaning_parameters"]
    cleaning_parallelization = step05["parallelization"]

    total_interactive_runtime = (
        step00["runtime_seconds"]
        + step01["runtime_seconds"]
        + step03["runtime_seconds"]
        + step04["runtime_seconds"]["interactive_setup"]
        + step05["runtime_seconds"]["interactive_setup"]
    )

    total_automated_runtime = (
        step02["runtime_seconds"]
        + step04["runtime_seconds"]["automated_runtime_seconds"]
        + step05["runtime_seconds"]["automated_mesh_cleaning"]
    )

    total_runtime = (
        total_interactive_runtime
        + total_automated_runtime
    )
    
    report_values = {
        "report_timestamp": timestamp,
        "report_timestamp_human":report_timestamp_human,

        "dataset_name": dataset_name,
        "scan_num": scan_num,
        "scanpath": step00["scanpath"],
        "slicepath": step00["slicepath"],
        "layoutfile": step00["layoutfile"],
        "layout_filename": os.path.basename(step00["layoutfile"]),
        "output_path": output_path,
        "metadata_path": step00["metadata_path"],

        "n_slices": step00["n_slices"],
        "first_slice": step00["first_slice"],
        "last_slice": step00["last_slice"],
        "first_slice_index": step00["first_slice_index"],
        "last_slice_index": step00["last_slice_index"],
        "slice_indices_are_consecutive": step00["slice_indices_are_consecutive"],
        "slice_index_fraction": step00["slice_index_fraction"],
        "voxel_size_mm": step00["voxel_size_mm"],
        "voxel_spacing_mm": step00["voxel_spacing_mm"],
        "is_isotropic": step00["is_isotropic"],
        "runtime_00": step00["runtime_seconds"],
        "runtime_00_formatted": format_runtime(step00["runtime_seconds"]),

        "transpose_preview": step01["transpose_preview"],
        "rotation_angle": step01["rotation_angle"],
        "rowrng": step01["rowrng"],
        "colrng": step01["colrng"],
        "runtime_01": step01["runtime_seconds"],
        "runtime_01_formatted": format_runtime(step01["runtime_seconds"]),

        "zwindow": step02["zwindow"],
        "remainder": step02["remainder"],
        "npz_file": step02["subvolume_file"],
        "reduced_volume_shape": step02["entire_subvolume_shape"],
        "subvolume_slice_shape": step02["subvolume_slice_shape"],
        "n_input_slices": step00["n_slices"],
        "runtime_02": step02["runtime_seconds"],
        "runtime_02_formatted": format_runtime(step02["runtime_seconds"]),

        "n_tiers_expected": step03.get("n_tiers_expected", "not recorded"),
        "n_active_tiers": step03.get("n_active_tiers", "not recorded"),
        "n_specimens_expected": step03["n_expected_specimens"],
        "n_specimens_identified": step03["n_extracted_specimens"],
        "n_extraction_regions": step03.get(
            "n_extraction_regions",
            step03["n_extracted_specimens"]
        ),
        "reverse_detected_tier_order": tier_segmentation["reverse_detected_tier_order"],
        "tier_detection_method": tier_segmentation["tier_detection_method"],
        "tier_boundaries": tier_segmentation["tier_boundaries"],
        "active_tier_ranges": tier_segmentation["active_tier_ranges"],
        "tier_rotation_summary": tier_normalization["tier_rotations"],
        "divider_summary": divider_review["tier_divider_definitions"],
        "extraction_plan_csv": step03["extraction_plan_csv"],
        "runtime_03": step03["runtime_seconds"],
        "runtime_03_formatted": format_runtime(step03["runtime_seconds"]),

        "n_specimens_extracted": step04.get("n_specimens_extracted", "not recorded"),
        "n_meshes_generated": step04.get("n_meshes_generated", "not recorded"),
        "n_surfacing_errors": step04["n_surfacing_errors"],
        "extract_num_cores": surfacing_parallelization["extract_num_cores"],
        "extract_num_cores_method": surfacing_parallelization["extract_num_cores_method"],
        "surface_num_cores": surfacing_parallelization["surface_num_cores"],
        "surface_num_cores_method": surfacing_parallelization["surface_num_cores_method"],
        "iso": surfacing_parameters["iso"],
        "iso_method": surfacing_parameters.get("iso_method", "not recorded"),
        "padding": surfacing_parameters["padding"],
        "mesh_folder": step04["mesh_folder"],
        "surfacing_errors_csv": step04["surfacing_errors_csv"],
        "extraction_runtime_seconds": step04["runtime_seconds"]["extraction"],
        "surfacing_runtime_seconds": step04["runtime_seconds"]["surfacing"],
        "runtime_04": step04["runtime_seconds"]["total_runtime_seconds"],
        "runtime_04_formatted": format_runtime(
            step04["runtime_seconds"]["total_runtime_seconds"]
        ),
        "interactive_setup_runtime_04": step04["runtime_seconds"]["interactive_setup"],
        "interactive_setup_runtime_04_formatted": format_runtime(step04["runtime_seconds"]["interactive_setup"]),
        "automated_runtime_04": step04["runtime_seconds"]["automated_runtime_seconds"],
        "automated_runtime_04_formatted": format_runtime(step04["runtime_seconds"]["automated_runtime_seconds"]),
        

        "clean_num_cores": cleaning_parallelization["num_cores"],
        "clean_num_cores_method": cleaning_parallelization["clean_num_cores_method"],
        "dust_cutoff": cleaning_parameters["dust_cutoff"],
        "hole_tolerance": cleaning_parameters["hole_tolerance"],
        "n_input_meshes": step05["n_input_meshes"],
        "n_meshes_cleaned": step05["n_meshes_cleaned"],
        "n_mesh_cleaning_failures": step05["n_mesh_cleaning_failures"],
        "mesh_cleaning_log_csv": step05["mesh_cleaning_log_csv"],
        "mesh_cleaning_failure_summary": (
            f"{step05['n_mesh_cleaning_failures']} mesh-cleaning failures recorded. "
            f"See mesh-cleaning log: {step05['mesh_cleaning_log_csv']}"
        ),
        "clean_mesh_folder": step05["clean_mesh_folder"],
        "interactive_setup_runtime_05": step05["runtime_seconds"]["interactive_setup"],
        "interactive_setup_runtime_05_formatted": format_runtime(step05["runtime_seconds"]["interactive_setup"]),
        "automated_mesh_cleaning_runtime_05": step05["runtime_seconds"]["automated_mesh_cleaning"],
        "automated_mesh_cleaning_runtime_05_formatted": format_runtime(step05["runtime_seconds"]["automated_mesh_cleaning"]),
        "runtime_05": step05["runtime_seconds"]["total_runtime_seconds"],
        "runtime_05_formatted": format_runtime(
            step05["runtime_seconds"]["total_runtime_seconds"]
        ),
        
        "total_interactive_runtime": total_interactive_runtime,
        "total_interactive_runtime_formatted": format_runtime(total_interactive_runtime),

        "total_automated_runtime": total_automated_runtime,
        "total_automated_runtime_formatted": format_runtime(total_automated_runtime),

        "total_runtime": total_runtime,
        "total_runtime_formatted": format_runtime(total_runtime),
    }

    markdown_text = render_text_template(template_path, report_values)

    with open(markdown_path, "w") as f:
        f.write(markdown_text)

    pdf_created = convert_markdown_text_to_pdf(markdown_text, pdf_path)

    return {
        "report_timestamp": timestamp,
        "report_timestamp_human": report_timestamp_human,
        "run_report_template_md": template_path,
        "run_report_markdown": markdown_path,
        "run_report_pdf": pdf_path if pdf_created else None,
        "pdf_created": pdf_created,
    }

        
def find_metadata_file_in_dataset(dataset_path):
    """
    Find the current workflow metadata file inside a dataset folder.
    Prefer metadata stored in an AMAAZE output folder.
    """

    dataset_path = normalize_path(dataset_path)

    matches = []

    for root, dirs, files in os.walk(dataset_path):
        for fname in files:
            if fname.endswith("_metadata.json"):
                matches.append(os.path.join(root, fname))

    if len(matches) == 0:
        raise RuntimeError(
            "No metadata file was found inside that dataset folder. "
            "Please run 00_share_data.py for this dataset before continuing."
        )

    amaaze_matches = [
        path for path in matches
        if "_AMAAZE_outputs" in os.path.basename(os.path.dirname(path))
    ]

    if len(amaaze_matches) == 1:
        return amaaze_matches[0]

    if len(amaaze_matches) > 1:
        amaaze_matches.sort(key=os.path.getmtime, reverse=True)
        return amaaze_matches[0]

    matches.sort(key=os.path.getmtime, reverse=True)
    return matches[0]
    
# ============================================================
# USER PROMPT HELPERS
# Used for interactive steps throughout the workflow.
# ============================================================

def ask(prompt, default=None, cast=str):
    """
    Ask the user a question and return the typed response.
    """

    if default is None:
        text = input(f"{prompt}\n> ").strip()
    else:
        text = input(f"{prompt}\n[{default}] > ").strip()
        if text == "":
            text = str(default)

    try:
        return cast(text)
    except ValueError:
        print(f"Could not understand '{text}'. Please try again.")
        return ask(prompt, default=default, cast=cast)


def ask_yes_no(prompt, default="y"):
    """
    Ask the user a yes/no question and return True or False.
    """

    default = default.lower()
    answer = input(f"{prompt} (y/n)\n[{default}] > ").strip().lower()

    if answer == "":
        answer = default

    if answer in ("y", "yes"):
        return True

    if answer in ("n", "no"):
        return False

    print("Please answer y or n.")
    return ask_yes_no(prompt, default=default)


def normalize_path(path_text):
    """
    Normalize a user-provided file or folder path.
    """

    return os.path.normpath(
        os.path.expanduser(
            path_text.strip().strip('"').strip("'")
        )
    )


def sanitize_name(name):
    """
    Convert user-provided text into a safe filename component.
    """

    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", str(name)).strip("_")

    if safe_name == "":
        raise RuntimeError("Dataset name cannot be empty after filename cleanup.")

    return safe_name
    
def ask_existing_path(prompt, is_dir=None):
    """
    Ask for a file or folder path until an existing path is provided.
    """

    while True:
        path = normalize_path(ask(prompt))

        if is_dir is True and os.path.isdir(path):
            return path

        if is_dir is False and os.path.isfile(path):
            return path

        if is_dir is None and os.path.exists(path):
            return path

        print()
        print("We could not find that path.")
        print("Please check for typos, missing folders, or copied quotation marks, then try again.")
        print()        

def ask_float_in_range(prompt, minimum, maximum, default=None):
    """
    Ask for a floating-point value within a specified range.
    """

    while True:
        value = ask(prompt, default=default, cast=float)

        if minimum <= value <= maximum:
            return value

        print()
        print(
            f"Please enter a value between {minimum} and {maximum}."
        )
        print()
          
    
# ============================================================
# 00_SHARE_DATA.PY HELPERS
# Metadata creation and initialization.
# ============================================================

def build_metadata_filename(dataset_name, scan_num):
    """
    Create a standardized metadata filename for a dataset and scan.
    """
    safe_dataset_name = sanitize_name(dataset_name)
    return f"{safe_dataset_name}_scan{scan_num}_metadata.json"

def create_output_folder(scanpath, dataset_name, scan_num):
    """
    Create and return the standardized AMAAZE output folder for this scan.
    """

    safe_dataset_name = sanitize_name(dataset_name)
    output_folder_name = f"{safe_dataset_name}_scan{scan_num}_AMAAZE_outputs"
    output_path = os.path.join(scanpath, output_folder_name)

    os.makedirs(output_path, exist_ok=True)

    return output_path
    
    
# ============================================================
# SHARED SLICE STACK HELPERS
# Used by 00, 01, 02, and 04.
# ============================================================

SUPPORTED_SLICE_EXTENSIONS = (".tif", ".tiff", ".dcm", ".dicom")


def extract_index(fname):
    """
    Return the last numeric index found in a filename.
    """

    basename = os.path.basename(fname)
    nums = re.findall(r"\d+", basename)

    if not nums:
        raise RuntimeError(f"No numeric index found in filename: {basename}")

    return int(nums[-1])


def get_sorted_slice_files(slicepath):
    """
    Find supported slice files and return them sorted by numeric filename index.
    """

    slicepath = normalize_path(slicepath)

    if not os.path.isdir(slicepath):
        raise RuntimeError(f"Slice folder not found: {slicepath}")

    slice_files = [
        os.path.join(slicepath, fname)
        for fname in os.listdir(slicepath)
        if fname.lower().endswith(SUPPORTED_SLICE_EXTENSIONS)
    ]

    if len(slice_files) == 0:
        raise RuntimeError(
            "No supported slice files found. Expected .tif, .tiff, .dcm, or .dicom."
        )

    indexed = [(fname, extract_index(fname)) for fname in slice_files]
    indices = [idx for _, idx in indexed]

    if len(indices) != len(set(indices)):
        raise RuntimeError("Duplicate numeric slice indices found in filenames.")

    indexed.sort(key=lambda x: x[1])

    sorted_files = [fname for fname, _ in indexed]
    sorted_indices = [idx for _, idx in indexed]

    return sorted_files, sorted_indices


def read_slice(slice_file):
    """
    Read one TIFF or DICOM slice and return it as an image array.
    """

    ext = os.path.splitext(slice_file)[1].lower()

    if ext in (".tif", ".tiff"):
        return io.imread(slice_file)

    if ext in (".dcm", ".dicom"):
        ds = pydicom.dcmread(slice_file)
        return ds.pixel_array

    raise RuntimeError(f"Unsupported slice file type: {slice_file}")
    
# ============================================================
# FUNCTIONS USED BY 01_SET_ROTATION_CROP.PY
# ============================================================ 

def apply_preview_orientation(im, transpose_preview):
    """
    Apply optional transpose so the preview image matches
    the specimen layout orientation.
    """

    if transpose_preview:
        im = im.T

    return im
    

def apply_preview_rotation(im, angle_deg):
    """
    Rotate preview image by the specified angle in degrees.
    """
    return rotate(
        im,
        angle_deg,
        resize=True,
        preserve_range=True
    )
    
    
def collect_crop_bounds(image):
    """
    Let the user click two opposite corners and return row/column crop bounds.
    """

    print()
    print("Click two opposite corners around the area you want to keep.")
    print("For example: upper-left and lower-right.")
    print("After selecting two corners, close the image window.")
    print("A crop preview will appear for confirmation.")
    print()

    fig, ax = plt.subplots()
    ax.imshow(image, cmap="gray")
    ax.set_title("Click two opposite crop corners, then close this window.")
    ax.axis("off")

    clicks = []

    def on_crop_corner_click(event):
        if event.inaxes != ax:
            return

        if event.xdata is None or event.ydata is None:
            return

        if len(clicks) >= 2:
            return

        x = int(round(event.xdata))
        y = int(round(event.ydata))

        clicks.append((x, y))
        ax.plot(x, y, "ro")
        fig.canvas.draw_idle()

        print(f"Crop corner {len(clicks)}/2: x={x}, y={y}")

    fig.canvas.mpl_connect("button_press_event", on_crop_corner_click)
    plt.show(block=False)
    
    input(
        "Click two crop corners in the image window. "
        "After both clicks appear, close the image window, then press Enter here..."
    )

    plt.close(fig)

    if len(clicks) != 2:
        print()
        print("Two crop corners were not selected.")
        print("Let's try again.")
        print()
        return None, None

    print()
    print("Crop selection received.")
    print("Opening crop preview...")
    print()
    
    xs = [pt[0] for pt in clicks]
    ys = [pt[1] for pt in clicks]

    colrng = [min(xs), max(xs)]
    rowrng = [min(ys), max(ys)]

    return rowrng, colrng
    
    
def update_preview(ax, fig, image, title):
    """
    redraw axes when preview updates to avoid distortion.
    """
    ax.clear()
    ax.imshow(image, cmap="gray", aspect="equal")
    ax.set_title(title)
    ax.axis("off")
    fig.canvas.draw_idle()
    plt.pause(0.1)
    
# ============================================================
# FUNCTIONS USED BY 02_build_volume.py
# ============================================================

# NOTE(dev):
# Aspect-ratio-preserving resize replaces the legacy fixed 225x225 resize.
# This avoids geometric distortion for rectangular scans while retaining
# computational reduction for downstream segmentation.
#
# IMPORTANT:
# Downstream scripts must not assume square XY dimensions.
#
# Current implementation preserves NumPy default float precision after
# z-window averaging and stores the reduced volume using np.savez().
#
# Future optimization questions:
# - Should reduced volumes use np.savez_compressed()?
# - Should imstack be explicitly cast to float32 or uint16?
# - What downstream effects would dtype reduction have on thresholding,
#   segmentation stability, marching cubes, and voxel interpretation?
#
# These questions should be evaluated empirically before optimization.

def resize_preserve_aspect(im, max_edge=225):
    h, w = im.shape[:2]

    if h >= w:
        new_h = max_edge
        new_w = max(1, round(w * max_edge / h))
    else:
        new_w = max_edge
        new_h = max(1, round(h * max_edge / w))

    return cv.resize(im, (new_w, new_h)).copy()


# ============================================================
# FUNCTIONS USED BY 03_segment.py
# ============================================================

def select_tier_boundaries_by_prominence(q, peaks, peak_props, n_tiers):
    """
    Select tier boundaries using peak prominence.

    Assumes:
    - the first detected peak represents the left/top box boundary,
    - len(q) represents the right/bottom scan boundary,
    - internal tier boundaries are selected from the most prominent internal peaks.
    """

    peaks = np.array(peaks).astype(int)
    prominences = np.array(peak_props["prominences"]).astype(float)

    if len(peaks) == 0:
        raise RuntimeError("No tier-boundary peaks detected.")

    n_internal_needed = n_tiers - 1

    left_boundary = int(peaks[0])
    right_boundary = int(len(q))

    internal_peaks = peaks[1:-1]
    internal_prominences = prominences[1:-1]

    if len(internal_peaks) < n_internal_needed:
        raise RuntimeError(
            f"Expected {n_internal_needed} internal tier boundaries, "
            f"but only found {len(internal_peaks)} internal peak candidates."
        )

    keep = np.argsort(internal_prominences)[-n_internal_needed:]
    selected_internal = np.sort(internal_peaks[keep])

    selected = np.concatenate((
        np.array([left_boundary]),
        selected_internal,
        np.array([right_boundary])
    ))

    return selected.astype(int)


def estimate_grid_rotation_by_coherence(im, angle_min=-5, angle_max=5, angle_step=0.25):
    """
    Estimate small rotation correction by testing candidate angles and choosing
    the angle that makes row/column divider structure most coherent.
    """

    angles = np.arange(angle_min, angle_max + angle_step, angle_step)

    scores = []

    for angle in angles:
        rotated = rotate(
            im,
            angle,
            preserve_range=True,
            resize=False,
            mode="edge"
        )

        # Ignore border artifacts during scoring.
        margin_r = max(1, int(rotated.shape[0] * 0.05))
        margin_c = max(1, int(rotated.shape[1] * 0.05))

        core = rotated[
            margin_r:rotated.shape[0] - margin_r,
            margin_c:rotated.shape[1] - margin_c
        ]

        scores.append(score_axis_coherence(core))

    scores = np.array(scores)
    best_idx = int(np.argmax(scores))
    best_angle = float(angles[best_idx])

    return best_angle, angles, scores


def local_stability_image(image, window_size=5, percentile_low=2, percentile_high=98):
    """
    Create a local-stability image.

    Low local standard deviation = high stability.
    Percentile stretch improves visual contrast.
    """

    image = image.astype(float)

    local_std = ndimage.generic_filter(
        image,
        np.std,
        size=window_size
    )

    stability = 1.0 / (local_std + 1)

    lo = np.percentile(stability, percentile_low)
    hi = np.percentile(stability, percentile_high)

    if hi > lo:
        stability = (stability - lo) / (hi - lo)
        stability = np.clip(stability, 0, 1)

    return stability
    

def paired_edge_centerlines(binary_mask, axis_label, min_fraction=0.45, min_pair_gap=2, max_pair_gap=20):
    """
    Find paired edge rows/columns in a binary mask and return midpoint centerlines.
    """

    if axis_label == "row":
        occupancy = binary_mask.mean(axis=1)
    elif axis_label == "col":
        occupancy = binary_mask.mean(axis=0)
    else:
        raise ValueError("axis_label must be 'row' or 'col'")

    candidate_idxs = np.where(occupancy >= min_fraction)[0]
    
    candidate_bands, band_centers = collapse_candidate_bands(
        candidate_idxs,
        max_band_gap=2
    )

    pairs = []
    centerlines = []

    used = set()

    for idx in band_centers:
        idx = float(idx)

        possible = band_centers[
            (band_centers >= idx + min_pair_gap) &
            (band_centers <= idx + max_pair_gap)
        ]

        if len(possible) == 0:
            continue

        partner = float(possible[0])

        pairs.append([float(idx), partner])
        centerlines.append(int(round((idx + partner) / 2)))

        used.add(int(idx))
        used.add(partner)

    return centerlines, pairs, occupancy, candidate_idxs, candidate_bands, band_centers


def review_dividers(
    image,
    proposed_rows=None,
    proposed_cols=None,
    title="Review divider locations"
):
    """
    Review, accept, add, or replace divider locations.

    ENTER with no clicks:
        accept proposed dividers

    Click:
        create a completely new divider set
    """

    if proposed_rows is None:
        proposed_rows = []

    if proposed_cols is None:
        proposed_cols = []

    fig, ax = plt.subplots()

    ax.imshow(image, cmap="gray")
    ax.set_title(title)

    for r in proposed_rows:
        ax.axhline(r, color="lime", linestyle="--")

    for c in proposed_cols:
        ax.axvline(c, color="lime", linestyle="--")

    clicked_rows = []
    clicked_cols = []

    def on_divider_review_click(event):

        if event.inaxes != ax:
            return

        if event.button == 1:
            row = int(round(event.ydata))
            clicked_rows.append(row)
            print(f"Row divider {len(clicked_rows)}: {row}")
            ax.axhline(row, color="red")
            fig.canvas.draw_idle()

        elif event.button == 3:
            col = int(round(event.xdata))
            clicked_cols.append(col)
            print(f"Column divider {len(clicked_cols)}: {col}")
            ax.axvline(col, color="blue")
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_divider_review_click)

    plt.show(block=False)

    input(
        "\nENTER = accept proposed dividers\n"
        "Left click = row divider\n"
        "Right click = column divider\n"
        "Press ENTER when finished.\n"
    )
    
    plt.close(fig)

    if len(clicked_rows) == 0:
        final_rows = np.array(proposed_rows).astype(int)
    else:
        final_rows = np.sort(np.array(clicked_rows).astype(int))

    if len(clicked_cols) == 0:
        final_cols = np.array(proposed_cols).astype(int)
    else:
        final_cols = np.sort(np.array(clicked_cols).astype(int))

    print("\nFinal divider selection:")
    print("Rows:", final_rows)
    print("Cols:", final_cols)

    return final_rows, final_cols


def score_axis_coherence(im):
    """
    Score how well bright divider-like structure aligns with image rows/columns.
    Higher score means stronger row/column coherence.
    """

    im = im.astype(float)

    if im.max() > im.min():
        im = (im - im.min()) / (im.max() - im.min())
    else:
        return 0

    row_profile = im.mean(axis=1)
    col_profile = im.mean(axis=0)

    row_score = row_profile.max() - row_profile.min()
    col_score = col_profile.max() - col_profile.min()

    row_sharpness = np.abs(np.diff(row_profile)).max()
    col_sharpness = np.abs(np.diff(col_profile)).max()

    return row_score + col_score + row_sharpness + col_sharpness


def collapse_candidate_bands(candidate_idxs, max_band_gap=2):
    """
    Collapse nearby candidate indices into edge bands.

    Example:
        [66, 67, 73] -> bands [[66, 67], [73]]
    """

    candidate_idxs = np.array(candidate_idxs).astype(int)

    if len(candidate_idxs) == 0:
        return [], np.array([])

    bands = []
    current_band = [int(candidate_idxs[0])]

    for idx in candidate_idxs[1:]:
        idx = int(idx)

        if idx - current_band[-1] <= max_band_gap:
            current_band.append(idx)
        else:
            bands.append(current_band)
            current_band = [idx]

    bands.append(current_band)

    band_centers = np.array([
        np.mean(band) for band in bands
    ])

    return bands, band_centers

def collect_tier_boundary_clicks(fig, ax):
    clicked_x = []

    def on_tier_boundary_click(event):
        if event.inaxes != ax:
            return
        if event.xdata is None:
            return

        x_click = event.xdata
        clicked_x.append(x_click)
        ax.axvline(x_click, color="red", linestyle="--")
        fig.canvas.draw_idle()
        print(f"Clicked tier boundary x = {x_click:.2f}")

    fig.canvas.mpl_connect("button_press_event", on_tier_boundary_click)
    return clicked_x
    
def remove_border_dividers(dividers, max_value, border_margin_fraction=0.03):
    dividers = np.array(dividers).astype(int)
    border_margin = int(max_value * border_margin_fraction)

    return dividers[
        (dividers > border_margin) &
        (dividers < max_value - border_margin)
    ]
    
# ============================================================
# FUNCTIONS USED BY 04_surface.py
# ============================================================
    
    
def estimate_iso_from_click_samples(image, min_clicks_per_class=3):
    """
    Estimate a baseline isovalue from user-selected air/background
    and specimen/material points.
    """

    def collect_iso_samples(label, color):
        print()
        print(f"Click at least {min_clicks_per_class} obvious {label} points.")
        print("Use points that clearly represent that category.")
        print("Close the window when finished, then press Enter in the terminal.")
        print()

        fig, ax = plt.subplots()
        ax.imshow(image, cmap="gray")
        ax.set_title(f"Click {label} points, then close this window")
        ax.axis("off")

        samples = []

        def on_iso_sample_click(event):
            if event.inaxes != ax:
                return
            if event.xdata is None or event.ydata is None:
                return

            x = int(round(event.xdata))
            y = int(round(event.ydata))
            value = float(image[y, x])

            samples.append({"x": x, "y": y, "value": value})
            ax.plot(x, y, "o", color=color)
            fig.canvas.draw_idle()

            print(f"{label} sample {len(samples)}: x={x}, y={y}, value={value}")

        fig.canvas.mpl_connect("button_press_event", on_iso_sample_click)
        plt.show(block=False)

        input(f"Press Enter after selecting {label} points and closing the window...")

        plt.close(fig)

        if len(samples) < min_clicks_per_class:
            raise RuntimeError(
                f"ISO helper needs at least {min_clicks_per_class} {label} samples."
            )

        return samples

    air_samples = collect_iso_samples("air/background", "cyan")
    specimen_samples = collect_iso_samples("specimen/material", "red")

    air_values = np.array([s["value"] for s in air_samples], dtype=float)
    specimen_values = np.array([s["value"] for s in specimen_samples], dtype=float)

    air_mean = float(np.mean(air_values))
    specimen_mean = float(np.mean(specimen_values))
    specimen_weight = 0.75

    iso_estimate = float(
        air_mean
        + specimen_weight * (specimen_mean - air_mean)
    )
    
    print()
    print(f"Average air/background grayscale value: {air_mean:.2f}")
    print(f"Average specimen/material grayscale value: {specimen_mean:.2f}")
    print(f"Estimated baseline ISO: {iso_estimate:.2f}")
    print()

    iso_metadata = {
        "method": "click_sample_specimen_weighted",
        "min_clicks_per_class": min_clicks_per_class,
        "air_background_samples": air_samples,
        "specimen_material_samples": specimen_samples,
        "air_background_mean": air_mean,
        "specimen_material_mean": specimen_mean,
        "specimen_weight": specimen_weight,
        "estimated_iso": iso_estimate,
    }

    return iso_estimate, iso_metadata
    
def prepare_iso_preview_image(
    slice_files,
    slice_index_fraction,
    transpose_preview,
    rotation_angle,
    rowrng,
    colrng
):
    """
    Prepare the representative slice used for ISO sampling.

    This uses the same orientation, rotation, and crop choices
    already stored in workflow metadata.
    """

    slice_index = int(len(slice_files) * slice_index_fraction)
    slice_index = min(slice_index, len(slice_files) - 1)

    raw_image = read_slice(slice_files[slice_index])

    preview_image = apply_preview_orientation(raw_image, transpose_preview)
    preview_image = rotate(
        preview_image,
        rotation_angle,
        preserve_range=True,
        resize=True
    )

    preview_image = preview_image[
        rowrng[0]:rowrng[1],
        colrng[0]:colrng[1]
    ].copy()

    return preview_image
    

def extract_specimen_subvolume_slice(
    slice_index,
    specimen_row,
    tier_z_start,
    image,
    outpath
):
    """
    Extract one specimen crop from one processed slice and append it
    to that specimen's temporary .npy stack.
    """

    specimen_id = str(specimen_row["specimen_id"])

    specimen_row_bounds = [
        int(specimen_row["row_start_padded"]),
        int(specimen_row["row_end_padded"])
    ]

    specimen_col_bounds = [
        int(specimen_row["col_start_padded"]),
        int(specimen_row["col_end_padded"])
    ]

    temp_filename = specimen_id + ".npy"

    with NpyAppendArray(
        os.path.join(outpath, temp_filename),
        delete_if_exists=(slice_index == tier_z_start)
    ) as npaa:
        npaa.append(
            image[
                specimen_row_bounds[0]:specimen_row_bounds[1],
                specimen_col_bounds[0]:specimen_col_bounds[1]
            ][None, :, :]
        )


# ============================================================
# FUNCTIONS USED BY 05_clean_meshes.py
# ============================================================        

def extract_max_subgraph(pts, tri):
    """Keep only the largest connected component of the mesh."""

    E = np.concatenate((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]), 0)

    I = E[:, 0] > E[:, 1]
    E[I, :] = E[I, -1::-1]

    E = E[E[:, 0] != E[:, 1], :]
    E = np.unique(E, axis=0)
    E = np.concatenate((E, E[:, -1::-1]), 0)

    n_pts = pts.shape[0]
    A = csr_matrix((np.ones(E.shape[0]), (E[:, 0], E[:, 1])), shape=(n_pts, n_pts))

    nseg, labs = connected_components(csgraph=A, directed=False, return_labels=True)

    counts = np.array([np.sum(labs == i) for i in range(nseg)])
    pt_ind2keep = np.where(labs == counts.argmax())[0]

    newind = np.arange(pt_ind2keep.shape[0])
    old2new = -1 * np.ones(n_pts, dtype=int)
    old2new[pt_ind2keep] = newind

    newtri = old2new[tri]
    newtri = newtri[np.sum(newtri < 0, 1) == 0, :]

    newpts = pts[pt_ind2keep, :]

    return newpts, newtri


def clean_mesh_file(fname, input_mesh_folder, clean_mesh_folder, dust_cutoff, hole_tolerance):
    """Clean one mesh and return a log record describing the cleaning decision."""

    input_mesh_path = os.path.join(input_mesh_folder, fname)
    output_mesh_path = os.path.join(clean_mesh_folder, fname)

    record = {
        "mesh_filename": fname,
        "input_mesh_path": input_mesh_path,
        "output_mesh_path": output_mesh_path,
        "status": "failure",
        "error_message": "",
        "dust_cutoff": dust_cutoff,
        "hole_tolerance": hole_tolerance,
        "n_components_found": None,
        "component_sizes": None,
        "n_components_above_dust_cutoff": None,
        "hole_detection_success": False,
        "hole_detection_error": "",
        "holes_by_candidate_component": None,
        "n_components_within_hole_tolerance": None,
        "selected_component_label": None,
        "selected_component_size": None,
        "selection_method": None,
    }

    try:
        M = tm.load_ply(input_mesh_path)

        try:
            holes = M.detect_holes()
            record["hole_detection_success"] = True
        except Exception as e:
            holes = None
            record["hole_detection_error"] = str(e)
            print(f"{fname}: detect_holes failed ({e}); falling back to largest-component cleanup only")

        ncomp, labs, counts = M.con_comp(returncounts=True)

        record["n_components_found"] = int(ncomp)
        record["component_sizes"] = counts.astype(int).tolist()

        print(f"{fname}: {len(counts)} components")

        labels_in_consideration = np.where(counts > dust_cutoff)[0]
        record["n_components_above_dust_cutoff"] = int(labels_in_consideration.shape[0])

        if labels_in_consideration.shape[0] > 0:
            counts_in_consideration = counts[labels_in_consideration]

            if holes is None:
                selected_label = int(labels_in_consideration[counts_in_consideration.argmax()])
                record["selection_method"] = "largest_component_above_dust_cutoff_hole_detection_failed"

            else:
                holes_in_each = []

                for i in labels_in_consideration:
                    ex = labs == i
                    holes_in_each.append(int(holes[ex].sum()))

                holes_in_each = np.array(holes_in_each)
                record["holes_by_candidate_component"] = holes_in_each.astype(int).tolist()

                keep = holes_in_each <= hole_tolerance
                record["n_components_within_hole_tolerance"] = int(keep.sum())

                labels_holeless = labels_in_consideration[keep]
                counts_holeless = counts_in_consideration[keep]

                if len(counts_holeless) == 0:
                    selected_label = int(counts.argmax())
                    record["selection_method"] = "largest_component_no_candidate_met_hole_tolerance"
                else:
                    selected_label = int(labels_holeless[counts_holeless.argmax()])
                    record["selection_method"] = "largest_component_within_hole_tolerance"

        else:
            print(f"{fname}: dust cutoff failed; extracting largest component")
            selected_label = int(counts.argmax())
            record["selection_method"] = "largest_component_no_component_above_dust_cutoff"

        record["selected_component_label"] = selected_label
        record["selected_component_size"] = int(counts[selected_label])

        M = M.extract_subtri(labs == selected_label)
        M.to_ply(output_mesh_path)

        record["status"] = "success"

    except Exception as e:
        record["status"] = "failure"
        record["error_message"] = str(e)
        print(f"{fname}: mesh cleaning failed ({e})")

    return record
