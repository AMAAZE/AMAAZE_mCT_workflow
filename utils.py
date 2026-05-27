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
import pydicom as dicom

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
# USER INPUT CONFIGURATION
# ============================================================
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    "--config",
    default=os.path.join(os.path.dirname(__file__), "user_inputs.json"),
    help="Path to the JSON config file for this scan."
)

args, _ = parser.parse_known_args()
CONFIG_PATH = os.path.normpath(args.config)

if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")

with open(CONFIG_PATH, "r") as f:
    CONFIG = json.load(f)

# Required global settings
try:
    scanpath = CONFIG["scanpath"]
    slicepath = CONFIG["slicepath"]
    scan_num = CONFIG["scan_num"]
    layoutfile = CONFIG["layoutfile"]
    n_dividers_row = CONFIG["n_dividers_row"]
    n_dividers_col = CONFIG["n_dividers_col"]
    zwindow = CONFIG["zwindow"]
    iso = CONFIG["iso"]
    padding = CONFIG["padding"]
    calibration_filename = CONFIG["calibration_filename"]
except KeyError as e:
    raise KeyError(f"Missing required config key: {e}")

# Required preview settings
try:
    slice_index_fraction = CONFIG["slice_index_fraction"]
    ang2rot = CONFIG["ang2rot"]
    rowrng = CONFIG["rowrng"]
    colrng = CONFIG["colrng"]
    transpose_preview = CONFIG["transpose_preview"]
except KeyError as e:
    raise KeyError(f"Missing required preview config key: {e}")
    
# Optional configuration

# Optional divider settings
row_dividers_override = CONFIG.get("row_dividers_override", None) # THIS IS LIKELY LEGACY AND MAY BE DELETABLE
col_dividers_override = CONFIG.get("col_dividers_override", None) # THIS IS LIKELY LEGACY AND MAY BE DELETABLE

# Optional threshold settings
iso_thresholds_override = CONFIG.get("iso_thresholds_override", None) # WE SEEM TO HAVE TWO ISO SETTINGS IN THE JSON. I'M NOT SURE HOW THEY INTERACT. REVIEW. THIS IS LIKELY LEGACY.
voxel_probe_n_clicks = CONFIG.get("voxel_probe_n_clicks", 5)

try:
    voxel_probe_n_clicks = int(voxel_probe_n_clicks)
except (TypeError, ValueError):
    raise ValueError(
        f"voxel_probe_n_clicks must be an integer, got: {voxel_probe_n_clicks}"
    )

if voxel_probe_n_clicks <= 0:
    raise ValueError(
        f"voxel_probe_n_clicks must be > 0, got: {voxel_probe_n_clicks}"
    )

# Optional voxel size setting
voxel_size_mm = CONFIG.get("voxel_size_mm", None) # voxel_size_mm = in-plane voxel size (X/Y)
voxel_spacing_mm = CONFIG.get("voxel_spacing_mm", None) # voxel_spacing_mm = slice spacing in Z

if voxel_size_mm is not None:
    try:
        voxel_size_mm = float(voxel_size_mm)
    except (TypeError, ValueError):
        raise ValueError(
            f"voxel_size_mm must be a number in millimeters, got: {voxel_size_mm}"
        )

    if voxel_size_mm <= 0:
        raise ValueError(
            f"voxel_size_mm must be > 0, got: {voxel_size_mm}"
        )
        
if voxel_spacing_mm is not None:
    try:
        voxel_spacing_mm = float(voxel_spacing_mm)
    except (TypeError, ValueError):
        raise ValueError(
            f"voxel_spacing_mm must be a number in millimeters, got: {voxel_spacing_mm}"
        )

    if voxel_spacing_mm <= 0:
        raise ValueError(
            f"voxel_spacing_mm must be > 0, got: {voxel_spacing_mm}"
        )

# Optional CPU settings
extract_ncores = CONFIG.get("extract_ncores", None)
surface_ncores = CONFIG.get("surface_ncores", None)
clean_ncores = CONFIG.get("clean_ncores", None)

for key, value in {
    "extract_ncores": extract_ncores,
    "surface_ncores": surface_ncores,
    "clean_ncores": clean_ncores,
}.items():
    if value is not None:
        try:
            value = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be an integer or null, got: {value}")

        if value <= 0:
            raise ValueError(f"{key} must be > 0, got: {value}")

        if key == "extract_ncores":
            extract_ncores = value
        elif key == "surface_ncores":
            surface_ncores = value
        elif key == "clean_ncores":
            clean_ncores = value

# Optional mesh cleanup settings
dust_cutoff = CONFIG.get("dust_cutoff", 1000)
hole_tolerance = CONFIG.get("hole_tolerance", 0)

# Optional processing settings
allow_slice_gaps = CONFIG.get("allow_slice_gaps", False)

# ============================================================
# FUNCTIONS USED BY 01_set_rotation_crop.py
# ============================================================

def extract_index(fname):
    """Return the last numeric index found in a filename."""
    basename = os.path.basename(fname)
    nums = re.findall(r'\d+', basename)
    if not nums:
        raise RuntimeError(f"No numeric index found in filename: {basename}")
    return int(nums[-1])


def apply_preview_orientation(im, transpose_preview):
    """Apply optional transpose for preview orientation."""
    if transpose_preview:
        im = im.T
    return im


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

def get_representative_slice_from_fraction(slicepath, slice_index_fraction):
    """Load a representative slice from the stack based on a fractional position."""
    slicepath = os.path.normpath(slicepath)

    tif_files = []
    for ext in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
        tif_files.extend(glob.glob(os.path.join(slicepath, ext)))

    tif_files = sorted(set(tif_files))

    if len(tif_files) == 0:
        raise RuntimeError("No .tif or .tiff files found in the specified slice folder.")

    tif_files_with_idx = [(f, extract_index(f)) for f in tif_files]
    tif_files_with_idx.sort(key=lambda x: x[1])

    tif_files = [f for f, _ in tif_files_with_idx]

    slice_index = int(len(tif_files) * slice_index_fraction)
    slice_index = min(slice_index, len(tif_files) - 1)

    slice_file = tif_files[slice_index]
    raw_im = io.imread(slice_file)

    return raw_im, slice_file, slice_index


def prepare_preview_probe_views(raw_im, transpose_preview, ang2rot, rowrng, colrng):
    """Prepare oriented and cropped image views for voxel selection."""
    oriented_im = apply_preview_orientation(raw_im, transpose_preview)

    rotated_display = rotate(
        oriented_im,
        ang2rot,
        preserve_range=True,
        resize=True
    )

    cropped_display = rotated_display[rowrng[0]:rowrng[1], colrng[0]:colrng[1]].copy()

    return oriented_im, cropped_display


def sample_true_voxel_from_preview_click(
    x_click, y_click,
    oriented_im,
    ang2rot,
    rowrng,
    colrng
):
    """Map a click in the display view back to the original image and return the voxel value."""
    # Convert from cropped-display coordinates to full rotated-image coordinates
    x_rot = x_click + colrng[0]
    y_rot = y_click + rowrng[0]

    h, w = oriented_im.shape[:2]
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0

    # Undo the display rotation
    theta = np.deg2rad(-ang2rot)

    x_shift = x_rot - cx
    y_shift = y_rot - cy

    x_orig = (np.cos(theta) * x_shift - np.sin(theta) * y_shift) + cx
    y_orig = (np.sin(theta) * x_shift + np.cos(theta) * y_shift) + cy

    x_orig = int(round(x_orig))
    y_orig = int(round(y_orig))

    if (
        x_orig < 0 or x_orig >= oriented_im.shape[1] or
        y_orig < 0 or y_orig >= oriented_im.shape[0]
    ):
        return None

    return oriented_im[y_orig, x_orig]


def collect_voxel_probe_clicks(
    display_im,
    oriented_im,
    ang2rot,
    rowrng,
    colrng,
    n_clicks=5,
    label="divider"
):
    """Collect voxel values from user clicks on the preview image."""
    fig, ax = plt.subplots()
    ax.imshow(display_im, cmap="gray")
    ax.set_title(f"Click {n_clicks} points on {label}")
    ax.axis("off")

    clicked_values = []

    def onclick(event):
        if event.inaxes != ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        if len(clicked_values) >= n_clicks:
            return

        x_click = int(round(event.xdata))
        y_click = int(round(event.ydata))

        val = sample_true_voxel_from_preview_click(
            x_click=x_click,
            y_click=y_click,
            oriented_im=oriented_im,
            ang2rot=ang2rot,
            rowrng=rowrng,
            colrng=colrng
        )

        if val is None:
            print("Click mapped outside image bounds. Try again.")
            return

        clicked_values.append(int(val))
        ax.plot(x_click, y_click, "ro")
        fig.canvas.draw_idle()

        print(
            f"{label} click {len(clicked_values)}/{n_clicks}: "
            f"value = {int(val)}"
        )

        if len(clicked_values) == n_clicks:
            print(f"Finished collecting {label} clicks: {clicked_values}")

    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show(block=False)
    input(f"Press Enter after selecting {n_clicks} {label} points...")

    if len(clicked_values) != n_clicks:
        raise RuntimeError(
            f"Expected {n_clicks} {label} clicks, got {len(clicked_values)}. "
            "Rerun this step."
        )

    return np.array(clicked_values).astype(int)


def robust_thresholds_from_probes(divider_vals, specimen_vals):
    """Compute T1, T2, T3 thresholds from clicked divider and specimen values."""
    divider_vals = np.sort(np.array(divider_vals).astype(int))
    specimen_vals = np.sort(np.array(specimen_vals).astype(int))

    if len(divider_vals) < 3:
        raise RuntimeError("Need at least 3 divider clicks to compute T1 and T2.")

    if len(specimen_vals) < 2:
        raise RuntimeError("Need at least 2 specimen clicks to compute T3.")

    t1 = divider_vals[1]
    t2 = divider_vals[-2]
    t3 = specimen_vals[1]

    print("Sorted divider values :", divider_vals.tolist())
    print("Sorted specimen values:", specimen_vals.tolist())
    print(f"Computed thresholds -> T1={t1}, T2={t2}, T3={t3}")

    if not (t1 < t2):
        raise RuntimeError(
            "Computed divider thresholds are not ordered correctly. "
            "Please re-click divider points."
        )

    if t3 <= t2:
        print(
            "Warning: T3 is not greater than T2. "
            "This may mean the clicked specimen/divider values overlap."
        )

    return np.array([t1, t2, t3]).astype(int)


def divider_range_from_probes(divider_vals):
    """Compute divider intensity range from clicked divider values."""
    divider_vals = np.sort(np.array(divider_vals).astype(int))

    if len(divider_vals) < 3:
        raise RuntimeError("Need at least 3 divider clicks to compute divider intensity range.")

    divider_low = divider_vals[1]
    divider_high = divider_vals[-2]

    print("Sorted divider values:", divider_vals.tolist())
    print(f"Computed divider intensity range: {divider_low} to {divider_high}")

    if not (divider_low < divider_high):
        raise RuntimeError(
            "Computed divider intensity range is not ordered correctly. "
            "Please re-click divider points."
        )

    return np.array([divider_low, divider_high]).astype(int)



def get_iso_thresholds_from_voxel_probe(
    slicepath,
    slice_index_fraction,
    transpose_preview,
    ang2rot,
    rowrng,
    colrng,
    n_clicks=5
):
    """Collect voxel samples from divider and specimen regions and compute T1, T2, T3 thresholds."""
    raw_im, slice_file, slice_index = get_representative_slice_from_fraction(
        slicepath,
        slice_index_fraction
    )

    oriented_im, display_im = prepare_preview_probe_views(
        raw_im=raw_im,
        transpose_preview=transpose_preview,
        ang2rot=ang2rot,
        rowrng=rowrng,
        colrng=colrng
    )

    print(f"Using representative slice for probing: {slice_file}")
    print(f"Slice index selected from slice_index_fraction: {slice_index}")

    divider_vals = collect_voxel_probe_clicks(
        display_im=display_im,
        oriented_im=oriented_im,
        ang2rot=ang2rot,
        rowrng=rowrng,
        colrng=colrng,
        n_clicks=n_clicks,
        label="divider"
    )

    specimen_vals = collect_voxel_probe_clicks(
        display_im=display_im,
        oriented_im=oriented_im,
        ang2rot=ang2rot,
        rowrng=rowrng,
        colrng=colrng,
        n_clicks=n_clicks,
        label="specimen"
    )

    t1t2t3 = robust_thresholds_from_probes(
        divider_vals=divider_vals,
        specimen_vals=specimen_vals
    )

    return t1t2t3

def get_iso_thresholds_from_image_probe(probe_image, n_clicks=5):
    """
    Collect divider/specimen threshold clicks from an already-prepared image.

    This experimental version is for clicking on an image generated from the
    reduced .npz volume, rather than clicking on the original TIFF stack.
    """

    display_im = probe_image
    oriented_im = probe_image

    divider_vals = collect_voxel_probe_clicks(
        display_im=display_im,
        oriented_im=oriented_im,
        ang2rot=0,
        rowrng=[0, probe_image.shape[0]],
        colrng=[0, probe_image.shape[1]],
        n_clicks=n_clicks,
        label="divider"
    )

    specimen_vals = collect_voxel_probe_clicks(
        display_im=display_im,
        oriented_im=oriented_im,
        ang2rot=0,
        rowrng=[0, probe_image.shape[0]],
        colrng=[0, probe_image.shape[1]],
        n_clicks=n_clicks,
        label="specimen"
    )

    t1t2t3 = robust_thresholds_from_probes(
        divider_vals=divider_vals,
        specimen_vals=specimen_vals
    )

    return t1t2t3

def get_divider_range_from_image_probe(probe_image, n_clicks=5):
    """
    Collect divider intensity clicks from an already-prepared image.

    Experimental replacement for T1/T2/T3 threshold selection.
    Uses only divider material to define the intensity range used for grid detection.
    """

    divider_vals = collect_voxel_probe_clicks(
        display_im=probe_image,
        oriented_im=probe_image,
        ang2rot=0,
        rowrng=[0, probe_image.shape[0]],
        colrng=[0, probe_image.shape[1]],
        n_clicks=n_clicks,
        label="divider"
    )

    return divider_range_from_probes(divider_vals)

def draw_boxes(im, row, col, title=''):
    """Overlay bounding boxes on an image for visual inspection."""
    fig, ax = plt.subplots()
    plt.axis('off')
    plt.imshow(im)
    
    for j in range(row.shape[0]):
        rect = patches.Rectangle(
            (col[j,0],row[j,0]), 
            col[j,1]-col[j,0], 
            row[j,1]-row[j,0],  
            linewidth=1, 
            edgecolor='r', 
            facecolor='none'
        )
        ax.add_patch(rect)
    plt.title(title)
    plt.show()

def collect_divider_line_clicks(image, n_lines, axis_label):
    """
    Collect divider line coordinates from user clicks.

    axis_label should be "row" for horizontal dividers
    or "col" for vertical dividers.
    """

    fig, ax = plt.subplots()
    ax.imshow(image, cmap="gray")
    ax.set_title(f"Click {n_lines} {axis_label} divider lines")
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")

    clicks = []

    def onclick(event):
        if event.inaxes != ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        if len(clicks) >= n_lines:
            return

        if axis_label == "row":
            coord = int(round(event.ydata))
            ax.axhline(coord, color="red", linestyle="--")
        elif axis_label == "col":
            coord = int(round(event.xdata))
            ax.axvline(coord, color="red", linestyle="--")
        else:
            raise ValueError("axis_label must be 'row' or 'col'")

        clicks.append(coord)
        fig.canvas.draw_idle()
        print(f"{axis_label} divider {len(clicks)}/{n_lines}: {coord}")

    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show(block=False)
    input(f"Press Enter after selecting {n_lines} {axis_label} dividers...")

    if len(clicks) != n_lines:
        raise RuntimeError(
            f"Expected {n_lines} {axis_label} divider clicks, got {len(clicks)}."
        )

    return np.sort(np.array(clicks).astype(int))

    
def ang_rot(im, plot=True, title=''):
    """Estimate a rotation angle that best aligns divider structure in an image."""
    
    incr = 0.1
    stop = 1
    angs = cat((np.arange(-10, -stop, incr), np.arange(stop, 10, incr)))

    n_rows, n_cols = im.shape[:2]
    irow, icol = np.meshgrid(
        np.arange(n_rows),
        np.arange(n_cols),
        indexing="ij"
    )

    mask = (
        (icol - n_cols / 2) ** 2
        + (irow - n_rows / 2) ** 2
        >= (min(n_rows, n_cols) / 2.25) ** 2
    )
    
    notmask = ~mask

    nim = im.copy() - im.min()
    nim = nim/nim.max()

    nim[mask] = 0 
    nim = gaussian(nim, sigma=3)
    
    f1 = np.array([[-1,1], [-1,1]])
    f2 = np.array([[1,-1], [-1,1]])

    vh_sums = []
    di_sums = []

    for a in angs:
        # Empirical offset retained from the original workflow.
        imi = rotate(rotate(nim, a + 27.2, preserve_range=True, order=1), -27.2, preserve_range=True, order=1) 

        vert = convolve(imi, f1.T)
        hor = convolve(imi, f1)
        diag = convolve(imi, f2)

        vh_sum = (np.sum(vert[notmask]**2) + np.sum(hor[notmask]**2))
        di_sum = np.sum(diag[notmask]**2)

        vh_sums.append(vh_sum)
        di_sums.append(di_sum)

    vh_sums = np.array(vh_sums)
    di_sums = np.array(di_sums)
    
    x = angs
    y = vh_sums/di_sums
    
    coeff = np.polyfit(angs, y, 2)
    yfit = coeff[0] * x**2 + coeff[1] * x + coeff[2]

    ang_out = -coeff[1] / (2*coeff[0])

    if plot:
        plt.figure()
        plt.plot(angs, y)
        plt.plot(angs, yfit)
        plt.title(title)
        plt.xlabel('rotation')
        plt.ylabel('detail signal')
        
    return ang_out


def auto_seg(signal, n_seg):
    """Find a threshold that separates a signal into the desired number of regions."""
    t = signal.max()
    m = 0.99

    n_groups = 0
    n_iter = 0
    max_iter = 1000
    
    while n_groups != n_seg:
        if n_iter >= max_iter:
            raise RuntimeError(
                f"auto_seg could not find {n_seg} regions within {max_iter} iterations. "
                "Check signal quality or segmentation parameters."
            )
        t = m * t
        I = signal > t
        L = label(I)
        n_groups = L[0].max()
        n_iter += 1
        
    return I

def read_hyperparameters(fnumber, directory='.'):
    """Read segmentation parameters from ctX_params.txt for a given scan."""
    fname = os.path.join(directory, f'ct{fnumber}_params.txt')

    with open(fname, 'r') as file:
        text = file.read()
            
    q = text.splitlines()
    
    vout = []
    for i in range(8):
        line = q[i]
        vout.append(literal_eval(line.split(' ')[1]))
        
    return vout


def get_vertseg_if_there(fnumber, directory='.'):
    """Return the saved vertical segmentation value from ctX_params.txt, if present."""
    fname = os.path.join(directory, f'ct{fnumber}_params.txt')
    str2lookfor = 'VERTSEG' 
    
    fname = 'ct'+str(fnumber)+'_params.txt'
            
    with open(fname, 'r') as file:
        text = file.read()
            
    q = text.splitlines()
        
    s = ''
    for s in q:
        if s[0:len(str2lookfor)] == str2lookfor: 
            break
        
    if s[0:len(str2lookfor)]!=str2lookfor:
        print('No vertical segmentation found. Please verify ctX_params.txt file.')
        return None
    
    
    words = s.split(' ')
    
    if len(words) < 2:
        print('No vertical segmentation found. Please verify ctX_params.txt file.')
        return None
    
    if len(words[1]) < 1:
        print('No vertical segmentation found. Please verify ctX_params.txt file.')
        return None
        
    return literal_eval(words[1])


def update_param(fnumber, str2lookfor, ex, directory='.'):
    """Write or update a parameter value in ctX_params.txt."""
    
    exstr = str(ex)
    eout = ''
    for c in exstr:
        if c != ' ':
            eout = eout + c
    exstr = eout
    
    fname = os.path.join(directory, f'ct{fnumber}_params.txt')
            
    with open(fname, 'r') as file:
        text = file.read()
            
    lst = text.split(str2lookfor) 
    
    if len(lst) == 2:
        pre, post = lst
    
        rbd = post.find('\n') 
        
        if rbd != -1 and post[rbd] == '\n':
            tail = post[rbd:]
        else:
            tail = '' 
        
        text = pre + str2lookfor + '= ' + exstr + tail
    else: 
        text = text + '\n' + str2lookfor + '= ' + exstr        

    with open(fname, 'w') as file:
        file.write(text)


# def id_cardboard(si, frac_thresh=0.75, t1=37000, t2=39000, t3=41000):
#    """Identify divider regions using intensity tresholding."""
#    x= np.sum((si > t1) * (si < t2), 0).astype(int) - np.sum(si > t3, 0).astype(int)
#    #x[x<0] = 0
#    t4 = np.floor(frac_thresh*x.max())
#    x[x<t4] = 0
#    
#    return x

def id_cardboard(si, frac_thresh=0.75, divider_low=37000, divider_high=39000):
    """Identify divider regions using a divider intensity range."""
    # Old T1/T2/T3 logic:
    # x = np.sum((si > t1) * (si < t2), 0).astype(int) - np.sum(si > t3, 0).astype(int)

    x = np.sum((si >= divider_low) & (si <= divider_high), 0).astype(int)

    t4 = np.floor(frac_thresh*x.max())
    x[x<t4] = 0
    
    return x


# def autorot2(si, frac_thresh=0.75, t1=37000, t2=39000, t3=41000, title=''):
#    """Estimate scan tilt from divider structure and return the rotation and rotated signal."""
#    x = np.sum((si > t1) * (si < t2), 0).astype(int) - np.sum(si > t3, 0).astype(int)
#    
#    x[x < 0] = 0
#    
#    t4 = np.floor(frac_thresh * x.max())
#    
#    x_thresh = (x > t4).astype(float)
#    
#    a = ang_rot(x_thresh, title=title)
#    xx = rotate(x, a, preserve_range=True)
#    
#    xx[xx < t4] = 0
#    
#    return a, xx

def autorot2(si, frac_thresh=0.75, divider_low=37000, divider_high=39000, title=''):
    """Estimate scan tilt from divider structure and return the rotation and rotated signal."""
    # Old T1/T2/T3 logic:
    # x = np.sum((si > t1) * (si < t2), 0).astype(int) - np.sum(si > t3, 0).astype(int)

    x = np.sum((si >= divider_low) & (si <= divider_high), 0).astype(int)
    
    x[x < 0] = 0
    
    t4 = np.floor(frac_thresh * x.max())
    
    x_thresh = (x > t4).astype(float)
    
    a = ang_rot(x_thresh, title=title)
    xx = rotate(x, a, preserve_range=True)
    
    xx[xx < t4] = 0
    
    return a, xx


def collect_divider_line_clicks_free(image, axis_label, title=""):
    """
    Collect any number of divider clicks from an image.

    User clicks divider positions directly on the image.
    Press Enter in the terminal when finished.

    axis_label:
        "row" -> horizontal divider selection
        "col" -> vertical divider selection
    """

    fig, ax = plt.subplots()

    ax.imshow(image, cmap="gray")
    ax.set_title(title)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")

    clicks = []

    def onclick(event):

        if event.inaxes != ax:
            return

        if event.xdata is None or event.ydata is None:
            return

        if axis_label == "row":
            coord = int(round(event.ydata))
            ax.axhline(coord, color="red", linestyle="--")

        elif axis_label == "col":
            coord = int(round(event.xdata))
            ax.axvline(coord, color="blue", linestyle="--")

        else:
            raise ValueError("axis_label must be 'row' or 'col'")

        clicks.append(coord)

        fig.canvas.draw_idle()

        print(f"{axis_label} divider {len(clicks)}: {coord}")

    fig.canvas.mpl_connect("button_press_event", onclick)

    plt.show(block=False)

    input("Press Enter when finished selecting dividers...")

    plt.close(fig)

    return np.sort(np.array(clicks).astype(int))


# ============================================================
# FUNCTIONS USED BY 04_surface.py
# ============================================================

def read_voxel_size(calibration_fname):
    """Reads voxel size (mm) from a calibration file"""
    str2lookfor = 'Optimum voxel size'  
                
    with open(calibration_fname, 'r') as file:
        text = file.read()
            
    q = text.splitlines()
        
    s = ''
    for s in q:
        if s[0:len(str2lookfor)] == str2lookfor: 
            break
        
    if s[0:len(str2lookfor)] != str2lookfor:
        raise RuntimeError(f"Could not find voxel size in calibration file: {calibration_fname}")
        
    words = s.split(' ')
        
    for i in range(len(words)):
        if words[i] == 'microns':
            i = i - 1
            break
        
    dx = float(words[i]) * (10**-3)
    return dx


def get_voxel_size_mm(calibration_fname=None, user_voxel_size_mm=None):
    """
    Resolve voxel size in mm with the following priority:
    1) user-provided JSON value
    2) calibration report
    3) raise clear error if neither is available
    """
    if user_voxel_size_mm is not None:
        print(f"Using voxel size from user_inputs.json: {user_voxel_size_mm} mm")
        return float(user_voxel_size_mm)

    if calibration_fname and os.path.exists(calibration_fname):
        dx = read_voxel_size(calibration_fname)
        print(f"Using voxel size from calibration file: {dx} mm")
        return dx

    raise RuntimeError(
        "Voxel size could not be determined. "
        "Provide 'voxel_size_mm' in user_inputs.json or supply a valid calibration file."
    )


def surfacing_subproc(filename, directory, iso_level, write_gif=False):
    """Create a .ply mesh (and optional .gif from one CT volume (.npz)."""
    print('Loading ' + filename + '...')
    M = np.load(os.path.join(directory, filename))
    I = M['I']
    dx = M['dx']
    dz = M['dz']
    
    # Rescale image to account for different voxel spacing in z and x/y.
    J = rescale(I.astype(float), (dz / dx, 1, 1), mode='constant')
    
    try: 
        verts, faces, normals, values = measure.marching_cubes(J, iso_level)

        # Rescale image to account for different voxel spacing in z and x/y.
        mesh = tm.mesh(dx * verts, faces)
    
        # Reverse triangle orientation because marching_cubes returns inward normals.
        mesh.flip_normals()
    
        #Write mesh to file
        mesh_filename = os.path.join(directory, filename[:-4]+'_iso%d'%iso_level)
        print('Saving mesh to '+ mesh_filename + '...')
        mesh.to_ply(mesh_filename + '.ply')
    
        if write_gif:
            mesh.to_gif(mesh_filename + '.gif')
        return '0'
    except Exception as error:
        print('surfacing error with ', filename, ': ', error)
        return filename
        

def surface_bones_parallel(directory, iso=2500, write_gif=False, error_fname='./surfacing_errors.csv', ncores='all'):
    """"Surface all .npz specimen volumes in a directory using parallel processing."""
    
    ddd = os.listdir(directory)
    
    fnames = []
    for f in ddd:
        if f.lower().endswith('.npz'):
            fnames.append(f)

    if len(fnames) == 0:
        raise RuntimeError(
            f"No .npz files found in {directory}. "
            "Surface extraction requires extracted specimen volumes from 04_surface.py input data."
        )
    
    if ncores == 'all':
        num_cores = multiprocessing.cpu_count()
    else:
        num_cores = max(1, min(int(ncores), multiprocessing.cpu_count()))

    errs = Parallel(n_jobs=num_cores)(
        delayed(surfacing_subproc)(f, directory, iso, write_gif) for f in fnames
    )
    
    errs = np.array(errs)
    errs = errs[errs != '0']
    
    if len(errs) == 0:
        print('No surfacing errors. No error CSV written.')
    else:
        print(f'There were {len(errs)} surfacing errors. Saving CSV to {error_fname}')              
        pd.DataFrame(errs).to_csv(error_fname,header=False, index=False)


def extract_subvolume_slice(i, j, infot, zrng, im, outpath):
    """Extract one specimen subvolume slice and append it to the output stack."""
    infoj = infot[j, :]
    fname = infoj[0] + ".npy"
    rowrng2 = infoj[3:5].astype(int)
    colrng2 = infoj[5:7].astype(int)

    with NpyAppendArray(
        os.path.join(outpath, fname), 
        delete_if_exists=(i == zrng[0])
    ) as npaa:
        npaa.append(
            (im[rowrng2[0]:rowrng2[1], 
            colrng2[0]:colrng2[1]].T)[None, :, :]
        )


# ============================================================
# FUNCTIONS USED BY 05_clean_meshes.py
# ============================================================

def extract_max_subgraph(pts, tri):
    """Keep only the largest connected component of the mesh."""
    E = np.concatenate((tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]) , 0 )
    
    # Order each edge so the larger index is second.
    I = E[:,0] > E[:,1]
    E[I,:] = E[I,-1::-1]
    
    E = E[ E[:,0] != E[:,1],:]
    
    E = np.unique(E, axis = 0)
    E = np.concatenate( (E,E[:,-1::-1]), 0) # Add reverse edges for undirected graph construction.
        
    n_pts = pts.shape[0]
    
    A = csr_matrix((np.ones(E.shape[0]), (E[:,0],E[:,1])), shape=(n_pts, n_pts))
    
    nseg, labs = connected_components(csgraph=A, directed=False, return_labels=True)
    
    counts = []
    for i in range(nseg):
        counts.append(np.sum(labs==i))
    counts = np.array(counts)
    
    pt_ind2keep =  np.where(labs == counts.argmax())[0]
    
    newind = np.arange(pt_ind2keep.shape[0])
    
    old2new = -1 * np.ones(n_pts, dtype=int)
    old2new[pt_ind2keep] = newind 
    
    newtri = old2new[tri]
    newtri = newtri[ np.sum(newtri<0,1) ==0, :]
    
    newpts = pts[pt_ind2keep,:]
    
    return newpts, newtri


def clean_mesh_file(fname, meshsubfolder, newmeshsubfolder):
    """Clean a mesh file by removing small components and preferring low-hole components when available."""
    dc = dust_cutoff
    ht = hole_tolerance

    M = tm.load_ply(os.path.join(meshsubfolder, fname))

    try:
        holes = M.detect_holes()
    except Exception as e:
        print(f"{fname}: detect_holes failed ({e}); falling back to largest-component cleanup only")
        holes = None

    ncomp, labs, counts = M.con_comp(returncounts=True)
    print(f"{fname}: {len(counts)} components")

    labels_in_consideration = np.where(counts > dc)[0]

    if labels_in_consideration.shape[0] > 0:
        counts_in_consideration = counts[labels_in_consideration]

        if holes is None:
            M = M.extract_subtri(
                labs == labels_in_consideration[counts_in_consideration.argmax()]
            )
        else:
            holes_in_each = []
            for i in labels_in_consideration:
                ex = labs == i
                holes_in_each.append(holes[ex].sum())

            holes_in_each = np.array(holes_in_each)
            l = holes_in_each <= ht

            labels_holeless = labels_in_consideration[l]
            counts_holeless = counts_in_consideration[l]

            if len(counts_holeless) == 0:
                M = M.extract_subtri(labs == counts.argmax())
            else:
                M = M.extract_subtri(
                    labs == labels_holeless[counts_holeless.argmax()]
                )
    else:
        print(f"{fname}: dust cutoff failed; still extracting largest component")
        M = M.extract_subtri(labs == counts.argmax())

    M.to_ply(os.path.join(newmeshsubfolder, fname))
    return
