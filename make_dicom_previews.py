"""
make_dicom_previews.py

To run script: 
    python make_dicom_previews.py --scanpath /path/to/scan

Purpose
-------
Generate a lightweight sequence of JPEG preview images from a DICOM (.dcm/.dicom)
slice stack for rapid visual inspection.

Rationale
---------
Many image viewers allow users to quickly scroll through sequential image files
(e.g., by holding the arrow keys), making TIFF or JPEG stacks convenient for
examining scan orientation, identifying anatomical regions, and locating slices
of interest. DICOM files often cannot be browsed this way in standard image
viewers.

This script creates a sequentially numbered JPEG stack that functions as a
"flipbook" representation of the scan while preserving the original DICOM data
for all scientific analyses.

Important
---------
The generated JPEG files are intended ONLY for visual inspection.

They are:
    - NOT used for segmentation or mesh generation.
    - NOT used for quantitative measurements.
    - NOT suitable for publication-quality figures.
    - NOT a replacement for the original DICOM data.

To minimize storage requirements and maximize browsing speed, each slice is:
    - normalized to an 8-bit grayscale display image, and
    - saved as a JPEG using lossy compression.

The original DICOM files remain the authoritative source for all downstream
processing and should never be discarded after preview generation.

Output
------
Creates:

    <dataset>/Slices_preview/

containing files of the form:

    slice_000001.jpg
    slice_000002.jpg
    slice_000003.jpg
    ...

These images are intended solely to provide a fast, video-like browsing
experience for navigating large CT datasets.
"""

import argparse
from pathlib import Path
import shutil

import numpy as np
import pydicom
from PIL import Image


def find_dicom_files(scanpath):
    """Find .dcm and .dicom files in the provided folder."""
    scanpath = Path(scanpath)

    dicom_files = []
    dicom_files.extend(scanpath.glob("*.dcm"))
    dicom_files.extend(scanpath.glob("*.DCM"))
    dicom_files.extend(scanpath.glob("*.dicom"))
    dicom_files.extend(scanpath.glob("*.DICOM"))

    return sorted(dicom_files)


def get_sort_key(filepath):
    """
    Return a sorting key for a DICOM file.

    Preferred order:
        1. InstanceNumber
        2. ImagePositionPatient z-coordinate
        3. filename
    """
    try:
        ds = pydicom.dcmread(filepath, stop_before_pixels=True)

        if hasattr(ds, "InstanceNumber"):
            return (0, int(ds.InstanceNumber))

        if hasattr(ds, "ImagePositionPatient"):
            return (1, float(ds.ImagePositionPatient[2]))

    except Exception:
        pass

    return (2, filepath.name)


def normalize_to_uint8(pixel_array):
    """
    Convert one DICOM pixel array to an 8-bit grayscale image.

    This uses per-slice normalization because the JPEGs are intended only for
    visual browsing, not quantitative analysis.
    """
    img = pixel_array.astype(np.float32)

    min_val = np.min(img)
    max_val = np.max(img)

    if max_val == min_val:
        return np.zeros(img.shape, dtype=np.uint8)

    img = (img - min_val) / (max_val - min_val)
    img = img * 255

    return img.astype(np.uint8)


def make_preview_images(scanpath, output_folder_name="Slices_preview", quality=92, force=False):
    scanpath = Path(scanpath)

    if not scanpath.exists():
        raise FileNotFoundError(f"Scan path does not exist: {scanpath}")

    if not scanpath.is_dir():
        raise NotADirectoryError(f"Scan path is not a folder: {scanpath}")

    dicom_files = find_dicom_files(scanpath)

    if not dicom_files:
        raise FileNotFoundError(
            f"No .dcm or .dicom files found in: {scanpath}"
        )

    dicom_files = sorted(dicom_files, key=get_sort_key)

    output_path = scanpath / output_folder_name

    if output_path.exists():
        if force:
            shutil.rmtree(output_path)
        else:
            raise FileExistsError(
                f"Output folder already exists:\n"
                f"    {output_path}\n\n"
                f"To overwrite it, rerun with:\n"
                f"    --force"
            )

    output_path.mkdir(parents=True, exist_ok=True)

    print()
    print("Generating DICOM preview JPEGs...")
    print(f"Input folder:  {scanpath}")
    print(f"Output folder: {output_path}")
    print(f"Files found:   {len(dicom_files)}")
    print()

    for idx, filepath in enumerate(dicom_files, start=1):
        ds = pydicom.dcmread(filepath)

        img8 = normalize_to_uint8(ds.pixel_array)

        outname = filepath.stem + ".jpg"
        outfile = output_path / outname

        Image.fromarray(img8).save(outfile, quality=quality)

        if idx == 1 or idx % 100 == 0 or idx == len(dicom_files):
            print(f"Saved {idx}/{len(dicom_files)}: {outname}")

    print()
    print("Preview generation complete.")
    print()
    print(f"Created {len(dicom_files)} JPEG preview images in:")
    print(f"    {output_path}")
    print()
    print(
        "These images are intended for rapid visual navigation only "
        "and are not used for downstream analyses."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Create JPEG preview images from a DICOM slice stack."
    )

    parser.add_argument(
        "--scanpath",
        required=True,
        help="Path to the folder containing .dcm/.dicom files.",
    )

    parser.add_argument(
        "--quality",
        type=int,
        default=92,
        help="JPEG quality, from 1 to 95. Default is 92.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing Slices_preview folder if it already exists.",
    )

    args = parser.parse_args()

    make_preview_images(
        scanpath=args.scanpath,
        quality=args.quality,
        force=args.force,
    )


if __name__ == "__main__":
    main()
