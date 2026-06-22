#!/usr/bin/env python3

"""
utils_backup.py

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
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Rectangle

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
            [
                "pandoc",
                "-V", "geometry:margin=1in",
                "-V", "fontsize=10pt",
                "-V", "linestretch=1.05",
                "--wrap=preserve",
                "-",
                "-o",
                pdf_path,
            ],
            input=markdown_text,
            text=True,
            check=True,
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
    dataset_folder_name = step00["dataset_folder_name"]

    timestamp = current_timestamp_for_filename()
    
    report_timestamp_human = datetime.datetime.now().strftime(
        "%B %d, %Y at %I:%M %p"
    )

    template_path = os.path.join(
        os.path.dirname(__file__),
        "final_run_report_template.md"
    )

    report_base = f"final_run_report_{dataset_folder_name}_{timestamp}"
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
        "report_timestamp_human": report_timestamp_human,

        "dataset_folder_name": dataset_folder_name,
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
        "total_slice_bytes": step00["total_slice_bytes"],
        "total_slice_gb": step00["total_slice_gb"],
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
    Find the workflow metadata file inside the standardized
    output folder for a dataset.
    """

    while True:

        dataset_path = normalize_path(dataset_path)

        output_folders = []

        for fname in os.listdir(dataset_path):
            candidate = os.path.join(dataset_path, fname)

            if (
                os.path.isdir(candidate)
                and fname.endswith("_outputs")
            ):
                output_folders.append(candidate)

        output_folders.sort()

        if len(output_folders) == 0:
            print()
            print("No output folder was found in that dataset folder.")
            print("Please run 00_share_data.py first, or check that you selected the correct dataset folder.")
            print()

            dataset_path = ask_existing_path(
                "What is the full path to the dataset folder you want to continue working on?",
                is_dir=True
            )
            continue

        if len(output_folders) > 1:
            print()
            print("More than one output folder was found in that dataset folder.")
            print("Please choose the output folder for the run you want to continue.")
            print()

            for i, folder in enumerate(output_folders, start=1):
                print(f"{i}. {folder}")

            choice = ask(
                "Enter the number of the output folder to use.",
                cast=int
            )

            if choice < 1 or choice > len(output_folders):
                print()
                print("That number is not in the list. Please try again.")
                print()
                continue

            output_folder = output_folders[choice - 1]

        else:
            output_folder = output_folders[0]

        metadata_files = [
            os.path.join(output_folder, fname)
            for fname in os.listdir(output_folder)
            if fname.endswith("_metadata.json")
        ]

        metadata_files.sort()

        if len(metadata_files) == 0:
            print()
            print("No workflow metadata file was found in this output folder.")
            print("Please run 00_share_data.py first, or check that the output folder has not been moved or renamed.")
            print()
            print(output_folder)
            print()

            dataset_path = ask_existing_path(
                "What is the full path to the dataset folder you want to continue working on?",
                is_dir=True
            )
            continue

        if len(metadata_files) > 1:
            print()
            print("More than one metadata file was found in this output folder.")
            print("There should usually be only one standardized metadata file per output folder.")
            print("Please choose the metadata file for the run you want to continue.")
            print()

            for i, metadata_file in enumerate(metadata_files, start=1):
                print(f"{i}. {metadata_file}")

            choice = ask(
                "Enter the number of the metadata file to use.",
                cast=int
            )

            if choice < 1 or choice > len(metadata_files):
                print()
                print("That number is not in the list. Please try again.")
                print()
                continue

            return metadata_files[choice - 1]

        return metadata_files[0]    
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
  
def ask_dataset_folder_name(prompt):
    """
    Ask for the dataset folder name.

    This is the name of the dataset directory only, not a full path.
    It is used throughout the workflow as the dataset identifier.
    """

    while True:

        dataset_folder_name = ask(prompt)

        if dataset_folder_name == "":
            print()
            print("Please enter a dataset folder name.")
            print()
            continue

        return dataset_folder_name
    
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

def build_metadata_filename(dataset_folder_name):
    """
    Create a standardized metadata filename for a dataset and scan.
    """
            
    return f"{dataset_folder_name}_metadata.json"


def create_output_folder(scanpath, dataset_folder_name):
    """
    Create and return the standardized output folder for this scan.
    """
    output_folder_name = f"{dataset_folder_name}_outputs"
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

def choose_rotation_angle_interactively(
    oriented_image,
    dataset_folder_name
):
    """
    Interactive rotation selector.
    Returns one final rotation angle in degrees.
    """

    rotation_angle = {"value": 0.0}

    image_h, image_w = oriented_image.shape[:2]
    display_size = int(np.ceil(np.sqrt(image_h**2 + image_w**2)))
    display_center = display_size / 2

    fig, ax = plt.subplots()
    fig.patch.set_facecolor("white")
    ax.set_facecolor("black")
    ax.set_box_aspect(1)
    plt.subplots_adjust(bottom=0.25)

    rotated_image = apply_preview_rotation(
        oriented_image,
        rotation_angle["value"]
    )

    ax.imshow(
        rotated_image, 
        cmap="gray", 
        aspect="equal",
        extent = [
            display_center - rotated_image.shape[1] / 2,
            display_center + rotated_image.shape[1] / 2,
            display_center + rotated_image.shape[0] / 2,
            display_center - rotated_image.shape[0] / 2,
        ],
        zorder=1
    )
    
    ax.set_xlim(0, display_size)
    ax.set_ylim(display_size, 0)
    ax.add_patch(
        Rectangle(
            (0, 0),
            display_size,
            display_size,
            facecolor="black",
            zorder=0
        )
    )

    ax.axhline(display_center, color="red", linewidth=0.8)
    ax.axvline(display_center, color="red", linewidth=0.8)
    ax.set_title(
        f"{dataset_folder_name}\n"
        f"Rotation = {rotation_angle['value']:.2f} degrees"
    )

    ax.axis("off")

    slider_ax = fig.add_axes([0.20, 0.10, 0.50, 0.03])
    rotation_slider = Slider(
        slider_ax,
        "Rotation",
        -360.0,
        360.0,
        valinit=0.0,
        valstep=0.1
    )

    def update_rotation(value):
        rotation_angle["value"] = float(value)

        rotated_image = apply_preview_rotation(
            oriented_image,
            rotation_angle["value"]
        )

        ax.clear()
        ax.set_facecolor("black")
        ax.imshow(
            rotated_image, 
            cmap="gray", 
            aspect="equal",
            extent = [
                display_center - rotated_image.shape[1] / 2,
                display_center + rotated_image.shape[1] / 2,
                display_center + rotated_image.shape[0] / 2,
                display_center - rotated_image.shape[0] / 2,
            ],
            zorder=1

        )

        ax.set_xlim(0, display_size)
        ax.set_ylim(display_size, 0)
        ax.add_patch(
            Rectangle(
                (0, 0),
                display_size,
                display_size,
                facecolor="black",
                zorder=0
            )
        )

        ax.axhline(display_center, color="red", linewidth=0.8, zorder=2)
        ax.axvline(display_center, color="red", linewidth=0.8, zorder=2)
        ax.set_title(
            f"{dataset_folder_name}\n"
            f"Rotation = {rotation_angle['value']:.2f} degrees"
        )
        ax.axis("off")
        fig.canvas.draw_idle()

    rotation_slider.on_changed(update_rotation)

    minus_ax = fig.add_axes([0.82, 0.085, 0.035, 0.05])
    minus_button = Button(minus_ax, "-")

    plus_ax = fig.add_axes([0.86, 0.085, 0.035, 0.05])
    plus_button = Button(plus_ax, "+")

    def decrease_rotation(event):
        rotation_slider.set_val(rotation_slider.val - 0.1)

    def increase_rotation(event):
        rotation_slider.set_val(rotation_slider.val + 0.1)

    minus_button.on_clicked(decrease_rotation)
    plus_button.on_clicked(increase_rotation)

    accept_ax = fig.add_axes([0.45, 0.005, 0.10, 0.04])
    accept_button = Button(accept_ax, "Accept")

    def accept_rotation(event):
        plt.close(fig)

    accept_button.on_clicked(accept_rotation)

    plt.show(block=True)

    return rotation_angle["value"]

    
    
def collect_crop_bounds(image, dataset_folder_name):
    """
    Let the user click two opposite corners and return row/column crop bounds.
    """

    print()
    print("Click two opposite corners around the area you want to keep.")
    print("For example: upper-left and lower-right.")
    print("After selecting two corners, close the image window.")
    print("A crop preview will automatically appear for confirmation.")
    print()

    fig, ax = plt.subplots()
    ax.imshow(image, cmap="gray")
    ax.set_title(
        f"{dataset_folder_name}\n"
        "Click two opposite crop corners, then close this window."
    )
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

        print()
        if len(clicks) == 1:
            print(f"First crop corner: x={x}, y={y}")
        else:
            print(f"Second crop corner: x={x}, y={y}")

    fig.canvas.mpl_connect("button_press_event", on_crop_corner_click)
    plt.show(block=False)
    
    input(
        "Click two crop corners in the image window. "
        "After both clicks appear, close the image window, then press Enter here...\n"
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
    
    
def update_preview(ax, fig, image, title, dataset_folder_name):
    """
    redraw axes when preview updates to avoid distortion.
    """
    ax.clear()
    ax.imshow(image, cmap="gray", aspect="equal")
    ax.set_title(f"{dataset_folder_name}\n{title}")
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


###########################################################

# ============================================================
# FUNCTIONS USED BY 03_segment.py for calculating the occupancy threshold
# ============================================================

def estimate_occupancy_threshold_by_peak_distance(
    occupancy,
    threshold_step=0.01,
    close_distance=10,
    min_prominence=0,
):
    """
    Estimate an occupancy threshold by lowering from the shortest retained
    prominence peak until a far-away new peak appears.

    Logic:
    1. Find occupancy peaks and their prominences.
    2. Keep the most prominent peaks.
    3. Start at the occupancy height of the weakest retained peak.
    4. Lower the threshold one step at a time.
    5. If newly admitted peaks are close to retained peaks, keep lowering.
    6. If a newly admitted peak is far from every retained peak, stop and
       return the previous threshold.
    """

    occupancy = np.asarray(occupancy).astype(float)

    peaks, peak_props = find_peaks(
        occupancy,
        prominence=min_prominence,
    )

    if len(peaks) == 0:
        return None, []

    prominences = np.asarray(peak_props["prominences"]).astype(float)
    peak_values = occupancy[peaks]

    sorted_prominences = np.sort(prominences)
    prominence_jumps = np.diff(sorted_prominences)

    largest_jump_index = int(np.argmax(prominence_jumps))
    prominence_cutoff = sorted_prominences[largest_jump_index + 1]

    keep_order = np.where(prominences >= prominence_cutoff)[0]

    retained_peaks = peaks[keep_order]
    retained_prominences = prominences[keep_order]
    retained_peak_values = peak_values[keep_order]

    if len(retained_peaks) == 0:
        return None, []

    starting_threshold = float(occupancy[retained_peaks[0]])
    previous_threshold = starting_threshold

    previous_visible_peaks = peaks[
        peak_values >= previous_threshold
    ]

    threshold = starting_threshold - threshold_step

    stop_reason = "reached bottom of curve"

    final_threshold = float(previous_threshold)

    while threshold >= np.min(occupancy):

        current_visible_peaks = peaks[
            peak_values >= threshold
        ]

        newly_visible_peaks = np.setdiff1d(
            current_visible_peaks,
            previous_visible_peaks
        )

        for new_peak in newly_visible_peaks:

            nearest_retained_distance = float(
                np.min(np.abs(retained_peaks - new_peak))
            )

            if nearest_retained_distance > close_distance:
                stop_reason = (
                    f"new peak at {int(new_peak)} was "
                    f"{nearest_retained_distance:.1f} pixels from nearest retained peak"
                )

                peak_summaries = []

                for peak, prom in zip(retained_peaks, retained_prominences):
                    peak_summaries.append({
                        "peak_index": int(peak),
                        "peak_value": float(occupancy[peak]),
                        "prominence": float(prom),
                        "status": "retained",
                    })

                peak_summaries.append({
                    "peak_index": int(new_peak),
                    "peak_value": float(occupancy[new_peak]),
                    "prominence": float(prominences[np.where(peaks == new_peak)[0][0]]),
                    "nearest_retained_distance": nearest_retained_distance,
                    "status": "stopped_threshold",
                    "stop_reason": stop_reason,
                })

                return float(previous_threshold), peak_summaries

        previous_visible_peaks = current_visible_peaks
        previous_threshold = threshold
        threshold -= threshold_step

    peak_summaries = []

    for peak, prom in zip(retained_peaks, retained_prominences):
        peak_summaries.append({
            "peak_index": int(peak),
            "peak_value": float(occupancy[peak]),
            "prominence": float(prom),
            "status": "retained",
        })

    return final_threshold, peak_summaries











































def estimate_occupancy_threshold_by_prominence_gap(
    occupancy,
    min_prominence=0,
    threshold_padding=0.01,
):
    """
    Estimate occupancy threshold from the natural gap in peak prominences.

    Strategy:
    1. Find local peaks in the occupancy curve.
    2. Sort peak prominences from low to high.
    3. Find the largest jump in prominence.
    4. Keep peaks above that jump as standout peaks.
    5. Set threshold just below the lowest occupancy value among standout peaks.
    """

    occupancy = np.asarray(occupancy).astype(float)

#######################################################################################

    # DEV TEST: raise curve floor before peak detection

    raw_peak_values = occupancy[
        find_peaks(occupancy, prominence=min_prominence)[0]
    ]

    height_floor = np.percentile(
        raw_peak_values,
        0
    )

    occupancy_for_peaks = occupancy.copy()
    occupancy_for_peaks[occupancy_for_peaks < height_floor] = height_floor

    pop_cutoff = float(height_floor)

#######################################################################################

    peaks, peak_props = find_peaks(
        occupancy_for_peaks,
        prominence=min_prominence,
    )

    peaks, peak_props = find_peaks(
        occupancy,
        prominence=min_prominence,
    )

    if len(peaks) == 0:
        return None, []

    prominences = np.asarray(
        peak_props["prominences"]
    ).astype(float)

    if len(prominences) == 1:
        threshold = float(
            max(0, occupancy[peaks[0]] - threshold_padding)
        )

        return threshold, [{
            "peak_index": int(peaks[0]),
            "peak_value": float(occupancy[peaks[0]]),
            "prominence": float(prominences[0]),
        }]

    sorted_prominences = np.sort(prominences)
    prominence_jumps = np.diff(sorted_prominences)

    largest_jump_index = int(np.argmax(prominence_jumps))

    prominence_cutoff = sorted_prominences[
        largest_jump_index + 1
    ]


    peak_values = occupancy[peaks]

    sorted_peak_values = np.sort(peak_values)
    peak_value_jumps = np.diff(sorted_peak_values)

    largest_height_jump_index = int(
        np.argmax(peak_value_jumps)
    )

    height_cutoff = sorted_peak_values[
        largest_height_jump_index + 1
    ]


    prominence_keep_mask = (
        prominences >= prominence_cutoff
    )

    height_keep_mask = (
        peak_values >= height_cutoff
    )

    #keep_mask = (prominence_keep_mask | height_keep_mask)
    keep_mask = prominence_keep_mask
    #keep_mask = height_keep_mask

    standout_peaks = peaks[keep_mask]
    standout_prominences = prominences[keep_mask]
    standout_peak_values = occupancy[standout_peaks]

    rejected_peak_values = occupancy[peaks[~keep_mask]]
    rejected_peaks = peaks[~keep_mask]
    rejected_prominences = prominences[~keep_mask]

    if len(rejected_peak_values) > 0:
        threshold = float(
            max(0, np.max(rejected_peak_values) - threshold_padding)
        )
    else:
        threshold = float(
            max(0, np.min(standout_peak_values) - threshold_padding)
        )

    peak_summaries = []

    rank_order = np.argsort(standout_prominences)[::-1]

    for rank, idx in enumerate(rank_order, start=1):
        peak_index = int(standout_peaks[idx])

        peak_summaries.append({
            "rank": int(rank),
            "peak_index": peak_index,
            "peak_value": float(occupancy[peak_index]),
            "prominence": float(standout_prominences[idx]),
            "status": "retained",
        })

    reject_order = np.argsort(rejected_prominences)[::-1]

    for rank, idx in enumerate(reject_order[:4], start=1):
        peak_index = int(rejected_peaks[idx])

        peak_summaries.append({
            "rank": int(rank),
            "peak_index": peak_index,
            "peak_value": float(occupancy[peak_index]),
            "prominence": float(rejected_prominences[idx]),
            "status": "rejected",
        })

    return threshold, peak_summaries, pop_cutoff, occupancy_for_peaks




















def occupancy_threshold_landscape(
    occupancy,
    threshold,
    max_band_gap=2,
):
    """
    Holding tank for all the descriptors at the band-level
    and the occupancy-threshold-wide level.
    """

    candidate_idxs = np.where(occupancy >= threshold)[0]

    candidate_bands, band_centers = collapse_candidate_bands(
        candidate_idxs,
        max_band_gap=max_band_gap
    )

    band_ranges = []
    band_n_candidates_list = []
    band_centers_list = []
    band_widths = []
    band_width_curve_fractions = []

    for band, band_center in zip(candidate_bands, band_centers):

        left_index = int(band[0])
        right_index = int(band[-1])

        band_range = [left_index, right_index]

        band_n_candidates = int(len(band))

        band_width = right_index - left_index + 1

        band_width_curve_fraction = (
            band_width / len(occupancy)
        )

        band_ranges.append(band_range)
        band_n_candidates_list.append(band_n_candidates)
        band_centers_list.append(float(band_center))
        band_widths.append(band_width)
        band_width_curve_fractions.append(
            band_width_curve_fraction
        )

    center_gaps_left = []
    center_gaps_right = []

    for i, band_center in enumerate(band_centers_list):

        #
        # Gap to nearest thing on the left
        #

        if i == 0:
            center_gap_left = float(
                band_center
            )
        else:
            previous_band_center = band_centers_list[i - 1]

            center_gap_left = float(
                band_center - previous_band_center
            )

        #
        # Gap to nearest thing on the right
        #

        if i == len(band_centers_list) - 1:
            center_gap_right = float(
                (len(occupancy) - 1) - band_center
            )
        else:
            next_band_center = band_centers_list[i + 1]

            center_gap_right = float(
                next_band_center - band_center
            )

        center_gaps_left.append(
            center_gap_left
        )

        center_gaps_right.append(
            center_gap_right
        )

    return {

        "threshold": float(threshold),

        "curve_length": int(len(occupancy)),

        "n_bands": int(len(candidate_bands)),

        "band_ranges":
            band_ranges,

        "band_n_candidates_list":
            band_n_candidates_list,

        "total_n_candidates":
            int(len(candidate_idxs)),

        "band_centers":
            band_centers_list,

        "band_widths":
            band_widths,

        "band_width_curve_fractions":
            band_width_curve_fractions,

        "center_gaps_left":
            center_gaps_left,

        "center_gaps_right":
            center_gaps_right,
    }



def build_initial_threshold_table(
    occupancy,
    threshold_step=0.05,
    max_band_gap=2,
):
    """
    Build a coarse occupancy-threshold table.

    Each row contains the full occupancy landscape
    at one threshold value.

    No comparisons are made here.
    No stability is assigned here.
    No threshold decisions are made here.
    """

    max_threshold = float(np.max(occupancy))
    min_threshold = float(np.min(occupancy))

    threshold_table = []

    threshold = max_threshold

    while threshold >= min_threshold:

        landscape = occupancy_threshold_landscape(
            occupancy=occupancy,
            threshold=threshold,
            max_band_gap=max_band_gap,
        )

        threshold_table.append(landscape)

        threshold -= threshold_step

    return threshold_table


def compare_threshold_table_rows(
    row_before,
    row_after,
):
    """
    Describe what changed between two neighboring threshold landscapes.

    This does not decide whether either row is good or bad.
    """

    n_bands_before = row_before["n_bands"]
    n_bands_after = row_after["n_bands"]

    total_n_candidates_before = sum(row_before["band_n_candidates_list"])
    total_n_candidates_after = sum(row_after["band_n_candidates_list"])

    total_band_width_before = sum(row_before["band_widths"])
    total_band_width_after = sum(row_after["band_widths"])

    return {
        "threshold_before": row_before["threshold"],
        "threshold_after": row_after["threshold"],

        "n_bands_before": n_bands_before,
        "n_bands_after": n_bands_after,
        "n_bands_change": n_bands_after - n_bands_before,

        "band_centers_before": row_before["band_centers"],
        "band_centers_after": row_after["band_centers"],

        "band_ranges_before": row_before["band_ranges"],
        "band_ranges_after": row_after["band_ranges"],

        "band_n_candidates_before": row_before["band_n_candidates_list"],
        "band_n_candidates_after": row_after["band_n_candidates_list"],

        "total_n_candidates_before": total_n_candidates_before,
        "total_n_candidates_after": total_n_candidates_after,
        "total_n_candidates_change": (
            total_n_candidates_after - total_n_candidates_before
        ),

        "band_widths_before": row_before["band_widths"],
        "band_widths_after": row_after["band_widths"],

        "total_band_width_before": total_band_width_before,
        "total_band_width_after": total_band_width_after,
        "total_band_width_change": (
            total_band_width_after - total_band_width_before
        ),

        "center_gaps_left_before": row_before["center_gaps_left"],
        "center_gaps_left_after": row_after["center_gaps_left"],

        "center_gaps_right_before": row_before["center_gaps_right"],
        "center_gaps_right_after": row_after["center_gaps_right"],
    }

def assess_threshold_landscape_plausibility(
    landscape,
    expected_n_dividers,
    close_gap_fraction=0.15,
    max_bands_per_divider=2,
):
    """
    Assess whether one threshold landscape is plausible for the expected
    number of dividers.

    A divider can be represented by either:
    - one band, or
    - two closely spaced bands.

    This helper does not choose a final threshold.
    """

    n_bands = landscape["n_bands"]
    band_centers = landscape["band_centers"]
    band_widths = landscape["band_widths"]
    band_n_candidates_list = landscape["band_n_candidates_list"]

    total_n_candidates = landscape["total_n_candidates"]

    n_singleton_bands = sum(
        n == 1
        for n in band_n_candidates_list
    )

    expected_min_bands = int(expected_n_dividers)
    expected_max_bands = int(expected_n_dividers * max_bands_per_divider)

    band_count_plausible = (
        expected_min_bands <= n_bands <= expected_max_bands
    )

    if n_bands == 0:
        return {
            "threshold": landscape["threshold"],
            "is_plausible": False,
            "reason": "no bands detected",
            "n_bands": n_bands,
            "expected_min_bands": expected_min_bands,
            "expected_max_bands": expected_max_bands,
            "band_centers": band_centers,
            "band_widths": band_widths,
            "band_n_candidates_list": band_n_candidates_list,
            "center_gaps": [],
            "close_center_gaps": [],
            "total_n_candidates": total_n_candidates,
            "n_singleton_bands": n_singleton_bands,
        }

    center_gaps = [
        float(band_centers[i + 1] - band_centers[i])
        for i in range(len(band_centers) - 1)
    ]

    curve_length = landscape["curve_length"]

    if len(center_gaps) > 0:
        close_gap_cutoff = close_gap_fraction * curve_length
    else:
        close_gap_cutoff = 0

    close_center_gaps = [
        gap for gap in center_gaps
        if gap <= close_gap_cutoff
    ]

    paired_band_possible = (
        n_bands > expected_n_dividers
        and len(close_center_gaps) > 0
    )

    if band_count_plausible:
        is_plausible = True
        reason = "band count fits expected single-band or paired-band divider representation"

    else:
        is_plausible = False
        reason = "band count does not fit expected divider representation"

    return {
        "threshold": landscape["threshold"],
        "is_plausible": is_plausible,
        "reason": reason,
        "n_bands": n_bands,
        "expected_min_bands": expected_min_bands,
        "expected_max_bands": expected_max_bands,
        "band_centers": band_centers,
        "band_widths": band_widths,
        "band_n_candidates_list": band_n_candidates_list,
        "center_gaps": center_gaps,
        "close_center_gaps": close_center_gaps,
        "paired_band_possible": paired_band_possible,
        "curve_length": curve_length,
        "close_gap_cutoff": close_gap_cutoff,
        "total_n_candidates": total_n_candidates,
        "n_singleton_bands": n_singleton_bands,
    }

def identify_plausible_regions(
    threshold_table,
    expected_n_dividers,
):
    plausibility_table = [
        assess_threshold_landscape_plausibility(
            landscape=row,
            expected_n_dividers=expected_n_dividers,
        )
        for row in threshold_table
    ]

    plausible_regions = []
    region_start_index = None

    for i, row in enumerate(plausibility_table):

        if row["is_plausible"] and region_start_index is None:
            region_start_index = i

        if (
            (not row["is_plausible"] or i == len(plausibility_table) - 1)
            and region_start_index is not None
        ):
            region_end_index = i - 1 if not row["is_plausible"] else i

            plausible_regions.append({
                "start_index": region_start_index,
                "end_index": region_end_index,
                "threshold_high": plausibility_table[region_start_index]["threshold"],
                "threshold_low": plausibility_table[region_end_index]["threshold"],
                "n_rows": region_end_index - region_start_index + 1,
                "rows": plausibility_table[region_start_index:region_end_index + 1],
            })

            region_start_index = None

    if len(plausible_regions) > 0:
        lowest_plausible_region = plausible_regions[-1]
    else:
        lowest_plausible_region = None

    return plausibility_table, plausible_regions, lowest_plausible_region


def identify_occupancy_threshold(
    occupancy,
    lowest_plausible_region,
    expected_n_dividers,
    fine_step=0.01,
    max_band_gap=2,
):
    """
    Choose an occupancy threshold from the lowest plausible coarse region.

    Fine pass rule:
    - prefer the lowest plausible threshold with zero singleton bands
    - if all plausible fine thresholds have singleton bands, use the lowest plausible threshold
    """

    if expected_n_dividers == 0:
        return 1.0

    if lowest_plausible_region is None:
        return None

    threshold_high = lowest_plausible_region["threshold_high"]
    threshold_low = lowest_plausible_region["threshold_low"]

    fine_threshold_table = []

    threshold = threshold_high

    while threshold >= threshold_low:
        fine_threshold_table.append(
            occupancy_threshold_landscape(
                occupancy=occupancy,
                threshold=threshold,
                max_band_gap=max_band_gap,
            )
        )

        threshold -= fine_step

    fine_plausibility_table, fine_plausible_regions, fine_lowest_plausible_region = identify_plausible_regions(
        fine_threshold_table,
        expected_n_dividers=expected_n_dividers,
    )

    plausible_rows = [
        row
        for row in fine_plausibility_table
        if row["is_plausible"]
    ]

    if len(plausible_rows) == 0:
        return None

    for row in reversed(plausible_rows):
        if row["n_singleton_bands"] == 0:
            return row["threshold"]

    return plausible_rows[-1]["threshold"]


def summarize_plausible_regions(
    plausible_regions,
    threshold_table,
):
    """
    Summarize each plausible region for comparison.

    No region selection is performed here.
    No threshold selection is performed here.
    """

    max_threshold = threshold_table[0]["threshold"]
    min_threshold = threshold_table[-1]["threshold"]

    region_summaries = []

    for region in plausible_regions:

        rows = region["rows"]

        mean_n_bands = float(np.mean([
            row["n_bands"]
            for row in rows
        ]))

        mean_total_candidates = float(np.mean([
            row["total_n_candidates"]
            for row in rows
        ]))

        mean_n_singleton_bands = float(np.mean([
            sum(
                n == 1
                for n in row["band_n_candidates_list"]
            )
            for row in rows
        ]))

        region_summaries.append({

            "threshold_high":
                region["threshold_high"],

            "threshold_low":
                region["threshold_low"],

            "n_rows":
                region["n_rows"],

            "distance_from_maximum":
                max_threshold - region["threshold_high"],

            "distance_from_minimum":
                region["threshold_low"] - min_threshold,

            "mean_n_bands":
                mean_n_bands,

            "mean_total_candidates":
                mean_total_candidates,

            "mean_n_singleton_bands":
                mean_n_singleton_bands,

            "rows":
                rows,
        })

    return region_summaries






###########################################################








# ============================================================
# FUNCTIONS USED BY 03_segment.py
# ============================================================

def select_tier_boundaries_by_prominence(mean_intensity_profile_z, peaks, peak_props, n_tiers): 
    """
    Select tier boundaries using peak prominence.

    Assumes:
    - the first detected peak represents the top box boundary, i.e. tier 1, left part of the curve,
    - len(q) represents the right/bottom scan boundary, q is stupid we will assign a new name mean intensity something z or somethng. 
    - internal tier boundaries are selected from the most prominent internal peaks.
    - the bottom of the box is the rightmost peak on the curve and one of the boundaries of the bottom tier. 
    """

    peaks = np.array(peaks).astype(int)
    prominences = np.array(peak_props["prominences"]).astype(float)

    if len(peaks) == 0:
        raise RuntimeError("No tier-boundary peaks detected.")

    n_internal_needed = n_tiers - 1

    # Original
    left_boundary = int(peaks[0])
    right_boundary = int(len(mean_intensity_profile_z))

    # New
    # left_boundary = 0
    # right_boundary = len(mean_intensity_profile_z)


    #internal_peaks = peaks[1:-1]
    #internal_prominences = prominences[1:-1]

    internal_peaks = peaks
    internal_prominences = prominences

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
###########################################################

def generate_tier_boundary_candidates(
    mean_intensity_profile_z,
    n_tiers
):
    """
    Generate candidate tier-boundary peaks and their boundary scores.
    """

    peaks, peak_props = find_peaks(
        mean_intensity_profile_z,
        prominence=0,
        width=0
    )

    boundary_scores = (
        peak_props["prominences"]
        * np.sqrt(peak_props["widths"])
    )

    rank_order = np.argsort(boundary_scores)[::-1]

    peaks = peaks[rank_order]
    boundary_scores = boundary_scores[rank_order]
    n_candidate_peaks = min(len(peaks), n_tiers + 1)

    candidate_peak_props = {}

    for key in peak_props:
        candidate_peak_props[key] = peak_props[key][rank_order][:n_candidate_peaks]

    candidate_peaks = peaks[:n_candidate_peaks]
    candidate_boundary_scores = boundary_scores[:n_candidate_peaks]

    return candidate_peaks, candidate_peak_props, candidate_boundary_scores

def estimate_jitter_tolerance(profile):
    step_sizes = np.abs(np.diff(profile))
    step_sizes = step_sizes[step_sizes > 0]

    if len(step_sizes) == 0:
        return np.finfo(float).eps

    sorted_steps = np.sort(step_sizes)
    step_jumps = np.diff(sorted_steps)

    if len(step_jumps) == 0:
        return sorted_steps[0]

    largest_jump_index = np.argmax(step_jumps)

    # Natural-gap estimate: largest value before the big jump.
    natural_gap_tolerance = sorted_steps[largest_jump_index]

    # Fallback estimate if no clear separation exists.
    fallback_tolerance = np.percentile(sorted_steps, 75)

    # For first prototype: use the natural gap when it is larger than fallback;
    # otherwise use the fallback.
    return max(natural_gap_tolerance, fallback_tolerance)

def get_monotonic_runs(profile):
    """
    Collapse a 1D profile into monotonic runs.

    Each run records the start/end indices, direction, and total magnitude
    of a coherent upward or downward movement.
    """

    profile = np.asarray(profile).astype(float)

    runs = []

    if len(profile) < 2:
        return runs

    run_start = 0
    direction = 0

    for i in range(1, len(profile)):
        step = profile[i] - profile[i - 1]
        step_direction = np.sign(step)

        if step_direction == 0:
            continue

        if direction == 0:
            direction = step_direction
            continue

        if step_direction != direction:
            run_end = i - 1

            runs.append({
                "start_index": int(run_start),
                "end_index": int(run_end),
                "start_value": float(profile[run_start]),
                "end_value": float(profile[run_end]),
                "direction": int(direction),
                "magnitude": float(abs(profile[run_end] - profile[run_start])),
            })

            run_start = i - 1
            direction = step_direction

    run_end = len(profile) - 1

    runs.append({
        "start_index": int(run_start),
        "end_index": int(run_end),
        "start_value": float(profile[run_start]),
        "end_value": float(profile[run_end]),
        "direction": int(direction),
        "magnitude": float(abs(profile[run_end] - profile[run_start])),
    })

    return runs

def estimate_run_magnitude_cutoff(runs):
    """
    Estimate the cutoff separating small jitter runs from large structural runs.
    """

    magnitudes = np.array([
        run["magnitude"]
        for run in runs
        if run["magnitude"] > 0
    ])

    if len(magnitudes) == 0:
        return np.finfo(float).eps

    if len(magnitudes) == 1:
        return magnitudes[0]

    sorted_magnitudes = np.sort(magnitudes)
    jumps = np.diff(sorted_magnitudes)

    if len(jumps) == 0:
        return sorted_magnitudes[0]

    largest_jump_index = np.argmax(jumps)

    natural_cutoff = sorted_magnitudes[largest_jump_index]
    fallback_cutoff = np.percentile(sorted_magnitudes, 75)

    return max(natural_cutoff, fallback_cutoff)


def find_left_edge(profile):
    """
    Find the dominant left package edge using monotonic-run magnitudes.

    Small back-and-forth runs are treated as jitter. The edge is the
    highest point reached before the first large downward departure.
    """

    profile = np.asarray(profile).astype(float)

    runs = get_monotonic_runs(profile)
    run_magnitude_cutoff = estimate_run_magnitude_cutoff(runs)

    left_edge_index = 0
    left_edge_value = profile[0]

    for run in runs:
        start_index = run["start_index"]
        end_index = run["end_index"]

        run_slice = profile[start_index:end_index + 1]
        local_max_offset = np.argmax(run_slice)
        local_max_index = start_index + local_max_offset
        local_max_value = profile[local_max_index]

        if local_max_value > left_edge_value:
            left_edge_value = local_max_value
            left_edge_index = local_max_index

        if (
            run["direction"] < 0
            and run["magnitude"] > run_magnitude_cutoff
        ):

            return int(left_edge_index)

        return int(left_edge_index)



def find_right_edge(profile):

    reversed_profile = np.asarray(profile)[::-1]

    reversed_left_edge_index = find_left_edge(reversed_profile)

    right_edge_index = (
        len(profile)
        - 1
        - reversed_left_edge_index
    )

    return right_edge_index

def remove_edge_adjacent_duplicate_candidates(
    left_edge,
    right_edge,
    candidate_peaks,
    candidate_boundary_scores,
    n_tiers,
):
    """
    Remove candidate peaks that sit implausibly close to an independently
    detected edge, as long as enough internal candidates remain.
    """

    candidate_peaks = np.asarray(candidate_peaks).astype(int)
    candidate_boundary_scores = np.asarray(candidate_boundary_scores).astype(float)

    n_internal_needed = n_tiers - 1

    if len(candidate_peaks) <= n_internal_needed:
        return candidate_peaks, candidate_boundary_scores

    provisional_boundaries = np.sort(
        np.concatenate((
            np.array([left_edge]),
            candidate_peaks,
            np.array([right_edge])
        ))
    )

    gaps = np.diff(provisional_boundaries)

    if len(gaps) == 0:
        return candidate_peaks, candidate_boundary_scores

    typical_gap = np.median(gaps)

    candidates_to_remove = []

    positive_gaps = gaps[gaps > 0]

    if len(positive_gaps) > 0:
        small_gap_cutoff = np.min(positive_gaps)
    else:
        small_gap_cutoff = typical_gap

    left_neighbor = candidate_peaks[np.argmin(np.abs(candidate_peaks - left_edge))]
    left_gap = abs(left_neighbor - left_edge)

    right_neighbor = candidate_peaks[np.argmin(np.abs(candidate_peaks - right_edge))]
    right_gap = abs(right_neighbor - right_edge)

    if left_gap <= small_gap_cutoff:
       candidates_to_remove.append(left_neighbor)

    if right_gap <= small_gap_cutoff:
       candidates_to_remove.append(right_neighbor)

    keep_mask = ~np.isin(candidate_peaks, candidates_to_remove)

    candidate_peaks_after_removal = candidate_peaks[keep_mask]
    candidate_scores_after_removal = candidate_boundary_scores[keep_mask]

    if len(candidate_peaks_after_removal) < n_internal_needed:
        print(
            "WARNING: Edge-adjacent candidate removal would leave too few "
            "internal candidates, so no edge-adjacent candidates were removed."
        )
        return candidate_peaks, candidate_boundary_scores

    return candidate_peaks_after_removal, candidate_scores_after_removal

def select_tier_boundaries_by_edge_and_score(
    mean_intensity_profile_z,
    candidate_peaks,
    candidate_peak_props,
    candidate_boundary_scores,
    n_tiers,
):
    """
    Select final tier boundaries using independently detected package edges
    and ranked internal candidate peaks.
    """

    profile = np.asarray(mean_intensity_profile_z)

    left_edge = find_left_edge(profile)
    right_edge = find_right_edge(profile)

    candidate_peaks = np.asarray(candidate_peaks).astype(int)
    candidate_boundary_scores = np.asarray(candidate_boundary_scores).astype(float)

    n_internal_needed = n_tiers - 1

    internal_peaks, internal_scores = remove_edge_adjacent_duplicate_candidates(
        left_edge=left_edge,
        right_edge=right_edge,
        candidate_peaks=candidate_peaks,
        candidate_boundary_scores=candidate_boundary_scores,
        n_tiers=n_tiers,
    )

    if len(internal_peaks) < n_internal_needed:
        print(
            f"WARNING: Expected {n_internal_needed} internal tier boundaries, "
            f"but only found {len(internal_peaks)} candidates. "
            "Continuing so the user can manually override if needed."
        )

    n_to_select = min(len(internal_peaks), n_internal_needed)

    rank_order = np.argsort(internal_scores)[::-1]
    selected_internal = internal_peaks[rank_order[:n_to_select]]
    selected_internal = np.sort(selected_internal)

    selected_boundaries = np.concatenate((
        np.array([left_edge]),
        selected_internal,
        np.array([right_edge])
    ))

    return {
    "selected_boundaries": selected_boundaries.astype(int),
    "left_edge": int(left_edge),
    "right_edge": int(right_edge),
    "selected_internal": selected_internal.astype(int),
    "candidate_peaks": candidate_peaks.astype(int),
    "internal_peaks_after_edge_removal": internal_peaks.astype(int),
    "n_internal_needed": int(n_internal_needed),
    "n_internal_selected": int(len(selected_internal)),
}


def select_tier_boundaries_by_edge_and_prominence(
    mean_intensity_profile_z,
    peaks,
    peak_props,
    n_tiers,
    edge_tolerance=5,
):
    """
    Determine package edges independently and select internal tier
    boundaries by prominence.

    Strategy
    --------
    1. Determine left and right package edges independently.
    2. Remove any find_peaks() candidates that duplicate those edges.
    3. Rank remaining candidates by prominence.
    4. Keep the strongest n_tiers - 1 internal dividers.
    5. Return [left_edge, internal..., right_edge].
    """

    profile = np.asarray(mean_intensity_profile_z)
    peaks = np.asarray(peaks).astype(int)
    prominences = np.asarray(
        peak_props["prominences"]
    ).astype(float)

    n_internal_needed = n_tiers - 1

    # --------------------------------------------------
    # Determine package edges independently.
    #
    # TODO:
    # Replace this placeholder with the final edge-finding
    # algorithm. For now we simply use the scan ends.
    # --------------------------------------------------

    left_edge = 0
    right_edge = len(profile)

    # --------------------------------------------------
    # Remove candidates that coincide with edges.
    # --------------------------------------------------

    keep_mask = (
        (np.abs(peaks - left_edge) > edge_tolerance)
        &
        (np.abs(peaks - right_edge) > edge_tolerance)
    )

    internal_peaks = peaks[keep_mask]
    internal_prominences = prominences[keep_mask]

    if len(internal_peaks) < n_internal_needed:
        raise RuntimeError(
            f"Expected at least {n_internal_needed} "
            f"internal candidates but found "
            f"{len(internal_peaks)}."
        )

    # --------------------------------------------------
    # Keep the strongest internal dividers.
    # --------------------------------------------------

    keep = np.argsort(
        internal_prominences
    )[-n_internal_needed:]

    selected_internal = np.sort(
        internal_peaks[keep]
    )

    return np.concatenate((
        np.array([left_edge]),
        selected_internal,
        np.array([right_edge]),
    )).astype(int)


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


def local_homogeneity_image(image, window_size=5, percentile_low=2, percentile_high=98):
    """
    Create a local-homogeneity image.

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

def compute_occupancy_profiles(binary_mask):
    """
    Compute row and column occupancy profiles from a binary mask.

    Occupancy is the fraction of True pixels in each row or column.
    """

    row_occupancy = binary_mask.mean(axis=1)
    col_occupancy = binary_mask.mean(axis=0)

    return row_occupancy, col_occupancy

def estimate_large_gap_cutoff_from_candidates(row_candidates, col_candidates):
    """
    Estimate the gap cutoff that separates within-divider edge spacing
    from between-neighborhood spacing using row and column candidates together.
    """

    all_gaps = []

    row_candidates = np.array(row_candidates).astype(int)
    col_candidates = np.array(col_candidates).astype(int)

    if len(row_candidates) > 1:
        all_gaps.extend(np.diff(np.sort(row_candidates)).tolist())

    if len(col_candidates) > 1:
        all_gaps.extend(np.diff(np.sort(col_candidates)).tolist())

    all_gaps = np.array([g for g in all_gaps if g > 0]).astype(float)

    print()
    print("POOLED GAP DIAGNOSTICS")
    print("row_candidates:")
    print(row_candidates)
    print("col_candidates:")
    print(col_candidates)
    print("pooled gaps:")
    print(all_gaps)

    if len(all_gaps) == 0:
        return None

    if len(all_gaps) == 1:
        return all_gaps[0] + 1

    sorted_gaps = np.sort(all_gaps)
    gap_jumps = np.diff(sorted_gaps)

    if len(gap_jumps) == 0:
        return sorted_gaps[-1] + 1

    jump_index = int(np.argmax(gap_jumps))

    small_gap_max = sorted_gaps[jump_index]
    large_gap_min = sorted_gaps[jump_index + 1]

    large_gap_cutoff = (small_gap_max + large_gap_min) / 2

    print("sorted pooled gaps:")
    print(sorted_gaps)
    print("gap jumps:")
    print(gap_jumps)
    print("small_gap_max:")
    print(small_gap_max)
    print("large_gap_min:")
    print(large_gap_min)
    print("large_gap_cutoff:")
    print(large_gap_cutoff)

    return large_gap_cutoff

def score_candidate_neighborhood(neighborhood, max_value=None):
    """
    Score whether a candidate neighborhood looks like a real divider.

    Higher score = stronger internal-divider evidence.
    Singleton/isolated neighborhoods and border-adjacent neighborhoods
    score lower.
    """

    neighborhood = np.array(neighborhood).astype(int)

    if len(neighborhood) == 0:
        return -999

    edge_sets, edge_centers = split_neighborhood_into_edge_sets(
        neighborhood,
        max_value=max_value
    )

    score = 0

    # Strongest evidence: the neighborhood can produce two edge sets.
    if len(edge_centers) >= 2:
        score += 10

        edge_width = abs(float(edge_centers[1]) - float(edge_centers[0]))

        # Very tiny "pairs" are often just noise.
        if edge_width >= 2:
            score += 5

    # More than one candidate is better than singleton noise.
    if len(neighborhood) > 1:
        score += len(neighborhood)
    else:
        score -= 10

    # Penalize neighborhoods near the image border.
    if max_value is not None:
        border_margin = int(max_value * 0.05)

        if neighborhood.min() <= border_margin:
            score -= 10

        if neighborhood.max() >= max_value - border_margin:
            score -= 10

    return score

def partition_candidate_neighborhoods(
    candidate_idxs,
    expected_n_dividers=None,
    large_gap_cutoff=None,
    max_value=None,
):
    """
    Partition candidate divider-edge indices into natural candidate neighborhoods.

    First, split candidates at large natural gaps. These neighborhoods are
    candidate divider-line regions, not the gaps themselves.

    If more neighborhoods are found than expected dividers, trim likely
    box-edge neighborhoods from the ends, favoring interior neighborhoods
    that contain valid paired edge evidence.
    """

    candidate_idxs = np.array(candidate_idxs).astype(int)

    if len(candidate_idxs) == 0:
        return []

    candidate_idxs = np.sort(candidate_idxs)

    if len(candidate_idxs) == 1:
        return [candidate_idxs]

    gaps = np.diff(candidate_idxs)

    print()
    print("partition diagnostics")
    print("candidate_idxs:")
    print(candidate_idxs)
    print("gaps:")
    print(gaps)

    if expected_n_dividers is None:
        return [candidate_idxs]

    expected_n_dividers = int(expected_n_dividers)

    if expected_n_dividers <= 0:
        return []

    if len(gaps) == 0:
        return [candidate_idxs]

    positive_gaps = gaps[gaps > 0]

    if len(positive_gaps) == 0:
        return [candidate_idxs]

    #median_gap = np.median(positive_gaps)
    #large_gap_cutoff = max(3, median_gap * 3)

    if large_gap_cutoff is None:
        median_gap = np.median(positive_gaps)
        large_gap_cutoff = max(3, median_gap * 3)

    split_gap_positions = np.where(gaps >= large_gap_cutoff)[0]

    print("large_gap_cutoff:")
    print(large_gap_cutoff)

    print("split_gap_positions:")
    print(split_gap_positions)

    if len(split_gap_positions) > 0:
        print("split_gap_sizes:")
        print(gaps[split_gap_positions])

    neighborhoods = []
    start = 0

    for gap_position in split_gap_positions:
        stop = gap_position + 1
        neighborhoods.append(candidate_idxs[start:stop])
        start = stop

    neighborhoods.append(candidate_idxs[start:])

    print("natural neighborhoods before adjudication:")

    for i, neighborhood in enumerate(neighborhoods, start=1):
        print(f"  neighborhood {i}: {neighborhood}")

    if len(neighborhoods) > expected_n_dividers:

        neighborhood_scores = [
            score_candidate_neighborhood(
                neighborhood,
                max_value=max_value
            )
            for neighborhood in neighborhoods
        ]

        print("neighborhood adjudication scores:")
        for i, (neighborhood, score) in enumerate(
            zip(neighborhoods, neighborhood_scores),
            start=1
        ):
            print(f"  neighborhood {i}: score={score}, values={neighborhood}")

        keep_order = np.argsort(neighborhood_scores)[-expected_n_dividers:]
        keep_order = np.sort(keep_order)

        neighborhoods = [
            neighborhoods[idx]
            for idx in keep_order
        ]

    print("final neighborhoods after adjudication:")

    for i, neighborhood in enumerate(neighborhoods, start=1):
        print(f"  neighborhood {i}: {neighborhood}")

    return neighborhoods

def trim_weak_end_from_neighborhood(neighborhood, max_value=None):
    neighborhood = np.array(neighborhood).astype(int)

    if len(neighborhood) < 4:
        return neighborhood

    gaps = np.diff(neighborhood)

    if len(gaps) == 0:
        return neighborhood

    largest_gap_index = int(np.argmax(gaps))
    largest_gap = gaps[largest_gap_index]

    other_gaps = np.delete(gaps, largest_gap_index)

    if len(other_gaps) == 0:
        return neighborhood

    if largest_gap <= np.max(other_gaps):
        return neighborhood

    left_side = neighborhood[:largest_gap_index + 1]
    right_side = neighborhood[largest_gap_index + 1:]

    left_is_singleton = len(left_side) == 1
    right_is_singleton = len(right_side) == 1

    left_is_border = False
    right_is_border = False

    if max_value is not None:
        border_margin = int(max_value * 0.05)
        left_is_border = left_side.min() <= border_margin
        right_is_border = right_side.max() >= max_value - border_margin

    if left_is_singleton and left_is_border:
        print(f"trimmed weak left side from neighborhood: {left_side}")
        return right_side

    if right_is_singleton and right_is_border:
        print(f"trimmed weak right side from neighborhood: {right_side}")
        return left_side

    return neighborhood


def split_neighborhood_into_edge_sets(neighborhood, max_value=None):
    neighborhood = np.array(neighborhood).astype(int)

    neighborhood = trim_weak_end_from_neighborhood(
        neighborhood,
        max_value=max_value
    )

    if len(neighborhood) == 0:
        return [], []

    if len(neighborhood) == 1:
        return [neighborhood.tolist()], np.array([float(neighborhood[0])])

    if len(neighborhood) == 2:
        edge_sets = [
            [int(neighborhood[0])],
            [int(neighborhood[1])]
        ]

    elif len(neighborhood) == 3:
        gaps = np.diff(neighborhood)

        if gaps[0] == gaps[1]:
            edge_sets = [
                [int(neighborhood[0]), int(neighborhood[1])],
                [int(neighborhood[1]), int(neighborhood[2])]
            ]
        elif gaps[0] > gaps[1]:
            edge_sets = [
                [int(neighborhood[0])],
                neighborhood[1:].tolist()
            ]
        else:
            edge_sets = [
                neighborhood[:2].tolist(),
                [int(neighborhood[2])]
            ]

    else:
        gaps = np.diff(neighborhood)
        split_position = int(np.argmax(gaps)) + 1

        edge_sets = [
            neighborhood[:split_position].tolist(),
            neighborhood[split_position:].tolist()
        ]

    edge_centers = np.array([
        np.mean(edge_set)
        for edge_set in edge_sets
    ])

    return edge_sets, edge_centers

from itertools import combinations











def select_best_nonoverlapping_pair_set(band_centers, expected_n_pairs=None, axis_length=None, return_score=False):
    band_centers = np.array(band_centers).astype(float)

    if expected_n_pairs is None:
        return []

    possible_pairs = []

    for i, j in combinations(range(len(band_centers)), 2):
        possible_pairs.append((i, j, band_centers[i], band_centers[j], band_centers[j] - band_centers[i]))

    best_pair_set = None
    best_score = np.inf

    for pair_set in combinations(possible_pairs, expected_n_pairs):

        used = []
        widths = []

        for i, j, b1, b2, width in pair_set:
            used.extend([i, j])
            widths.append(width)

        if len(used) != len(set(used)):
            continue

        widths = np.array(widths).astype(float)

        mean_width = np.mean(widths)
        width_variation = np.std(widths)

        if axis_length is not None:

            centerlines = np.sort([
                (b1 + b2) / 2
                for i, j, b1, b2, width in pair_set
            ])

            boundaries = np.concatenate((
                [0],
                centerlines,
                [axis_length - 1]
            ))

            cell_widths = np.diff(boundaries)

            mean_cell_width = np.mean(cell_widths)
            cell_width_variation = np.std(cell_widths)

        else:

            mean_cell_width = 0
            cell_width_variation = 0

        score = (
           mean_width
           + (2 * width_variation)
           + (2 * cell_width_variation)
        )

        if score < best_score:
            best_score = score
            best_pair_set = pair_set

    if best_pair_set is None:
        if return_score:
            return [], np.inf
        return []

    pairs = [
        [float(b1), float(b2)]
        for i, j, b1, b2, width in best_pair_set
    ]

    if return_score:
        return pairs, best_score

    return pairs






def generate_split_band_center_scenarios(candidate_bands, expected_n_pairs):

    """
    Generate possible edge-center scenarios when too few band centers were found.

    If the layout expects N divider pairs, we need 2N edge centers.
    If fewer than 2N band centers were detected, this helper tries every
    possible way of splitting existing band centers into synthetic A/B edges.

    Example:
        detected: [B1, B2, B3]
        needed:   4 edge centers

        scenarios:
            [B1a, B1b, B2,  B3]
            [B1,  B2a, B2b, B3]
            [B1,  B2,  B3a, B3b]
    """

    if expected_n_pairs is None:
        return [np.array([np.mean(band) for band in candidate_bands]).astype(float)]

    expected_n_edges = int(expected_n_pairs) * 2
    current_n_edges = len(candidate_bands)

    base_centers = [float(np.mean(band)) for band in candidate_bands]

    if current_n_edges >= expected_n_edges:
        return [np.array(base_centers).astype(float)]

    n_splits_needed = expected_n_edges - current_n_edges

    if n_splits_needed > len(candidate_bands):
        return [np.array(base_centers).astype(float)]

    scenarios = []

    for split_indices in combinations(range(len(candidate_bands)), n_splits_needed):
        scenario = []

        for idx, band in enumerate(candidate_bands):
            band = np.array(band).astype(float)

            if idx in split_indices:
                if len(band) == 1:
                    scenario.extend([band[0] - 1, band[0] + 1])
                elif len(band) == 2:
                    scenario.extend([band[0], band[1]])
                elif len(band) == 3:
                    scenario.extend([band[0], band[-1]])
                else:
                    midpoint = len(band) // 2
                    scenario.extend([
                        float(np.mean(band[:midpoint])),
                        float(np.mean(band[midpoint:]))
                    ])
            else:
                scenario.append(float(np.mean(band)))

        scenarios.append(np.array(sorted(scenario)).astype(float))

    print()
    print("Generated scenarios:")

    for i, scenario in enumerate(scenarios):
        print(f"Scenario {i+1}: {scenario}")

    return scenarios

  
def paired_edge_centerlines(binary_mask, axis_label, min_fraction=0.45, min_pair_gap=2, max_pair_gap=20, expected_n_pairs=None):
    """
    Find paired edge rows/columns in a binary mask and return midpoint centerlines.
    """

    row_occupancy, col_occupancy = compute_occupancy_profiles(
        binary_mask
    )

    if axis_label == "row":
        occupancy = row_occupancy
    elif axis_label == "col":
        occupancy = col_occupancy
    else:
        raise ValueError("axis_label must be 'row' or 'col'")

    candidate_idxs = np.where(occupancy >= min_fraction)[0]
    
    candidate_bands, band_centers = collapse_candidate_bands(
        candidate_idxs,
        max_band_gap=2
    )

#######################

    edge_center_scenarios = generate_split_band_center_scenarios(
        candidate_bands,
        expected_n_pairs=expected_n_pairs
    )

    best_pairs = []
    best_score = np.inf
    best_scenario = None

    for scenario in edge_center_scenarios:

        pairs, score = select_best_nonoverlapping_pair_set(
            scenario,
            expected_n_pairs=expected_n_pairs,
            return_score=True
        )

        if len(pairs) == 0:
            continue

        if score < best_score:
            best_score = score
            best_pairs = pairs
            best_scenario = scenario

    pairs = best_pairs

    centerlines = [
        int(round((pair[0] + pair[1]) / 2))
        for pair in pairs
    ]

##############################

#    pairs = select_best_nonoverlapping_pair_set(
#        band_centers,
#        expected_n_pairs=expected_n_pairs
#    )

    print()
    print(f"{axis_label.upper()} DIAGNOSTICS")
    print("band_centers:", band_centers)
    print("expected_n_pairs:", expected_n_pairs)
    print("selected_pairs:", pairs)

#    centerlines = [
#        int(round((pair[0] + pair[1]) / 2))
#        for pair in pairs
#    ]
#
#    pairs = []
#    centerlines = []
#
#    used = set()
#
#    for idx in band_centers:
#        idx = float(idx)
#
#        possible = band_centers[
#            (band_centers >= idx + min_pair_gap) &
#            (band_centers <= idx + max_pair_gap)
#        ]
#
#        if len(possible) == 0:
#            continue
#
#       partner = float(possible[0])
#
#        pairs.append([float(idx), partner])
#        centerlines.append(int(round((idx + partner) / 2)))
#
#        used.add(int(idx))
#        used.add(partner)

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
    ax.set_title(f"{dataset_folder_name}\n{title}")

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