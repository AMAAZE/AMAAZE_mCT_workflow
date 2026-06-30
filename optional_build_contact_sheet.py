#!/usr/bin/env python3

"""
optional_build_contact_sheet.py

Author: Katrina E. Yezzi-Woodley

Build a PDF contact sheet from using outputs from optional_render_views.py

Layout:
    Specimen label
    image image image image

    Specimen label
    image image image image
    ...
"""

# ============================================================
# Configuration and imports
# ============================================================

from utils import *

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    raise RuntimeError(
        "Pillow is required to build the contact sheet.\n"
        "Install with: pip install Pillow\n"
        "Or skip this step if you do not need a contact sheet."
    )

# ============================================================
# Load workflow metadata
# ============================================================

TOTAL_QUESTIONS_cs = 1

print_terminal_header("Optional Script: Build contact sheet")

print("In this optional step, you will build a PDF contact sheet")
print("from PNG views created by optional_render_views.py.")
print()
print("This script writes a separate contact-sheet PDF.")
print("It does not modify the canonical workflow metadata JSON.")
print()
print("When you're ready, press Enter to begin.")
input("> ")

print_question_header("Workflow Metadata JSON", 1, TOTAL_QUESTIONS_cs)

metadata_paths = get_metadata_paths_from_command_line_or_user(
    step_name="optional_build_contact_sheet",
    allow_batch=False
)

metadata_path = metadata_paths[0]
metadata = load_metadata_if_available(metadata_path)

md = unpack_metadata(metadata)

# ============================================================
# Locate input and output files
# ============================================================

render_view_folders = [
    os.path.join(md.output_path, folder)
    for folder in os.listdir(md.output_path)
    if (
        os.path.isdir(os.path.join(md.output_path, folder))
        and folder.lower().startswith("optional_render_views")
    )
]

render_view_folders.sort()

if len(render_view_folders) == 0:
    raise RuntimeError(
        f"No optional render-view folder was found in:\n{md.output_path}\n\n"
        "Run optional_render_views.py first, then run this script again."
    )

if len(render_view_folders) == 1:
    photodir = os.path.normpath(render_view_folders[0])
else:
    print()
    print("More than one optional render-view folder was found.")
    print("Please choose the rendered PNG folder to use.")
    print()

    for i, folder in enumerate(render_view_folders, start=1):
        print(f"{i}. {folder}")

    print()

    choice = ask(
        "Enter the number of the render-view folder to use.",
        cast=int
    )

    if choice < 1 or choice > len(render_view_folders):
        raise RuntimeError("That number is not in the list.")

    photodir = os.path.normpath(render_view_folders[choice - 1])

timestamp = current_timestamp_for_filename()

outfile = os.path.join(
    md.output_path,
    f"optional_contact_sheet_{md.dataset_folder_name}_{timestamp}.pdf"
)

pngs = [f for f in os.listdir(photodir) if f.lower().endswith(".png")]

if len(pngs) == 0:
    raise RuntimeError(
        f"No PNG files found in:\n{photodir}\n\n"
        "Run optional_render_views.py first, then run this script again."
    )

print()
print(f"Found {len(pngs)} rendered PNG file(s).")
print(f"Render-view folder: {photodir}")
print(f"Contact sheet will be written to: {outfile}")
print()

# ============================================================
# Group rendered images by specimen
# ============================================================

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


# ============================================================
# Page layout settings
# ============================================================

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
IMG_H = int(IMG_W * 0.85)

BLOCK_H = LABEL_HEIGHT + LABEL_GAP + IMG_H + ROW_GAP


# ============================================================
# Contact sheet helpers
# ============================================================

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

# ============================================================
# Build contact sheet PDF
# ============================================================

pages = []
page = make_blank_page()
draw = ImageDraw.Draw(page)

scan_name = md.dataset_folder_name

try:
    title_font = ImageFont.truetype("arial.ttf", 48)
except Exception:
    title_font = ImageFont.load_default()

draw.text((LEFT, TOP), scan_name, fill="black", font=title_font)

y = TOP + 80

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

print_success("Contact sheet PDF created.")

print_step_complete_header("Optional contact sheet complete")

print(f"Contact sheet PDF:")
print()
print(f"    {outfile}")
print()
print(f"Specimens included: {len(specimen_names)}")
print(f"Pages written: {len(pages)}")
print()
