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
import time
import sys
import argparse
import multiprocessing
import timeit
import glob
import re

import numpy as np
import pandas as pd
import scipy
import scipy.stats as stats
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

import cv2 as cv
import skimage.io as io
import pydicom

# NOTE:
# Duplicate AMAAZE dicom imports are currently present.
# This should be consolidated in a future cleanup.
import amaazetools.trimesh as tm
from amaazetools.dicom import *
from amaazetools import dicom

from ast import literal_eval
from joblib import Parallel, delayed

from numpy import concatenate as cat
from npy_append_array import NpyAppendArray
from matplotlib import patches
from skimage import measure
from skimage.filters import gaussian
from skimage.transform import rotate, rescale
from scipy.ndimage import convolve, label
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
# FUNCTIONS USED BY 03_segment.py
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
    iso_estimate = float((air_mean + specimen_mean) / 2)

    print()
    print(f"Average air/background grayscale value: {air_mean:.2f}")
    print(f"Average specimen/material grayscale value: {specimen_mean:.2f}")
    print(f"Estimated baseline ISO: {iso_estimate:.2f}")
    print()

    iso_metadata = {
        "method": "click_sample_midpoint",
        "min_clicks_per_class": min_clicks_per_class,
        "air_background_samples": air_samples,
        "specimen_material_samples": specimen_samples,
        "air_background_mean": air_mean,
        "specimen_material_mean": specimen_mean,
        "estimated_iso": iso_estimate,
    }

    return iso_estimate, iso_metadata
