#!/usr/bin/env python3

"""
EX6_contact_sheet_pdf.py

Author: Katrina E. Yezzi-Woodley

Build a PDF contact sheet from EX5 specimen view images.

Expected input:
    <scanpath>/scanphotos/
        specimenA_view1.png
        specimenA_view2.png
        specimenA_view3.png
        specimenA_view4.png
        specimenB_view1.png
        ...

Output:
    <scanpath>/scanphotos/contact_sheet.pdf

Layout:
    Specimen label
    image image image image

    Specimen label
    image image image image
    ...
"""

from utils import *

import os
import re
from PIL import Image, ImageDraw, ImageFont


# --------------------------------------------------
# Paths
# --------------------------------------------------
photodir = os.path.normpath(os.path.join(scanpath, "scanphotos"))
outfile = os.path.join(photodir, "contact_sheet.pdf")

if not os.path.isdir(photodir):
    raise RuntimeError(
        f"scanphotos folder not found: {photodir}\n"
        "Run EX5 first."
    )

pngs = [f for f in os.listdir(photodir) if f.lower().endswith(".png")]

if len(pngs) == 0:
    raise RuntimeError(
        f"No PNG files found in {photodir}\n"
        "Run image render first."
    )


# --------------------------------------------------
# Group images by specimen stem
# Expects names like:
#   specimen_label_view1.png
#   specimen_label_view2.png
#   specimen_label_view3.png
#   specimen_label_view4.png
# --------------------------------------------------
pattern = re.compile(r"^(.*)_view([1-4])\.png$", re.IGNORECASE)

groups = {}
for fname in pngs:
    m = pattern.match(fname)
    if m is None:
        continue

    stem = m.group(1)
    view_num = int(m.group(2))

    if stem not in groups:
        groups[stem] = {}

    groups[stem][view_num] = os.path.join(photodir, fname)

if len(groups) == 0:
    raise RuntimeError(
        "No view images were found.\n"
        "Expected filenames like specimen_view1.png ... specimen_view4.png"
    )

# Keep only specimens with at least one image, sorted by label
specimen_names = sorted(groups.keys())


# --------------------------------------------------
# Page settings
# --------------------------------------------------
# PDF page size in pixels at working resolution
PAGE_W = 2550   # ~8.5 in at 300 dpi
PAGE_H = 3300   # ~11 in at 300 dpi

LEFT = 120
RIGHT = 120
TOP = 120
BOTTOM = 120

LABEL_HEIGHT = 60
LABEL_GAP = 20
ROW_GAP = 80
IMAGE_GAP = 20

usable_w = PAGE_W - LEFT - RIGHT
usable_h = PAGE_H - TOP - BOTTOM

# Four images across the usable width
IMG_W = (usable_w - 3 * IMAGE_GAP) // 4
IMG_H = int(IMG_W * 0.85)   # adjust later if you want a different visual ratio

BLOCK_H = LABEL_HEIGHT + LABEL_GAP + IMG_H + ROW_GAP


# --------------------------------------------------
# Font
# --------------------------------------------------
try:
    font = ImageFont.truetype("arial.ttf", 36)
except Exception:
    font = ImageFont.load_default()


def make_blank_page():
    return Image.new("RGB", (PAGE_W, PAGE_H), "white")


def add_specimen_block(page, y, specimen_label, image_paths):
    """
    Draw one specimen label plus up to 4 images in a row.
    """
    draw = ImageDraw.Draw(page)

    # Label
    draw.text((LEFT, y), specimen_label, fill="black", font=font)

    # Images start below label
    y_img = y + LABEL_HEIGHT + LABEL_GAP

    for i in range(4):
        x = LEFT + i * (IMG_W + IMAGE_GAP)

        if (i + 1) in image_paths:
            img = Image.open(image_paths[i + 1]).convert("RGB")
            img.thumbnail((IMG_W, IMG_H))

            # center image inside its slot
            paste_x = x + (IMG_W - img.width) // 2
            paste_y = y_img + (IMG_H - img.height) // 2
            page.paste(img, (paste_x, paste_y))
        else:
            # Optional placeholder border for missing views
            draw.rectangle(
                [x, y_img, x + IMG_W, y_img + IMG_H],
                outline="gray",
                width=2
            )
            draw.text(
                (x + 20, y_img + 20),
                f"missing view {i+1}",
                fill="gray",
                font=font
            )

    return y + BLOCK_H


# --------------------------------------------------
# Build pages
# --------------------------------------------------
pages = []
page = make_blank_page()
draw = ImageDraw.Draw(page)

scan_name = os.path.basename(os.path.normpath(scanpath))

# Title font (slightly larger than specimen labels)
try:
    title_font = ImageFont.truetype("arial.ttf", 48)
except Exception:
    title_font = ImageFont.load_default()

draw.text((LEFT, TOP), scan_name, fill="black", font=title_font)

y = TOP + 80   # shift content down below title

for specimen_label in specimen_names:
    image_paths = groups[specimen_label]

    # New page if next block will overflow
    if y + BLOCK_H > PAGE_H - BOTTOM:
        pages.append(page)
        page = make_blank_page()
        y = TOP

    y = add_specimen_block(page, y, specimen_label, image_paths)

# Append final page
pages.append(page)

# Save multipage PDF
pages[0].save(
    outfile,
    save_all=True,
    append_images=pages[1:],
    resolution=300.0
)

print(f"Saved contact sheet PDF to: {outfile}")
print(f"Specimens included: {len(specimen_names)}")
print(f"Pages written: {len(pages)}")
