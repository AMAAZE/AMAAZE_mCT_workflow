#!/usr/bin/env python3

"""
optional_build_gifs.py

Optional accessory utility for the AMAAZE mCT workflow.

Builds animated GIFs from PNG views created by optional_render_views.py.

This script writes separate GIF outputs.
It does not modify the canonical workflow metadata JSON.
"""

# ============================================================
# Configuration and imports
# ============================================================

from utils import *

try:
    from PIL import Image
except ImportError:
    raise RuntimeError(
        "Pillow is required to build GIFs.\n"
        "Install with: pip install Pillow\n"
        "Or skip this step if you do not need GIFs."
    )

# ============================================================
# Load workflow metadata
# ============================================================

TOTAL_QUESTIONS_gif = 2

print_terminal_header("Optional Script: Build GIFs")

print("In this optional step, you will build animated GIFs")
print("from PNG views created by optional_render_views.py.")
print()
print("This script writes separate GIF outputs.")
print("It does not modify the canonical workflow metadata JSON.")
print()
print("When you're ready, press Enter to begin.")
input("> ")

print_question_header("Workflow Metadata JSON", 1, TOTAL_QUESTIONS_gif)

metadata_paths = get_metadata_paths_from_command_line_or_user(
    step_name="optional_build_gifs",
    allow_batch=False
)

metadata_path = metadata_paths[0]
metadata = load_metadata_if_available(metadata_path)

md = unpack_metadata(metadata)

# ============================================================
# Locate input and output folders
# ============================================================

print_question_header("Render-view folder", 2, TOTAL_QUESTIONS_gif)

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

gif_outdir = os.path.join(
    md.output_path,
    f"optional_gifs_{md.dataset_folder_name}_{timestamp}"
)

os.makedirs(gif_outdir, exist_ok=True)

print()
print("Using render-view folder:")
print(f"    {photodir}")
print()
print("GIFs will be written to:")
print(f"    {gif_outdir}")
print()

# ============================================================
# Group rendered images by specimen
# ============================================================

pattern = re.compile(r"^(.*)_view([1-4])\.png$", re.IGNORECASE)

pngs = [
    fname for fname in os.listdir(photodir)
    if fname.lower().endswith(".png")
]

groups = {}

for fname in pngs:
    match = pattern.match(fname)

    if match is None:
        continue

    specimen_label = match.group(1)
    view_num = int(match.group(2))

    if specimen_label not in groups:
        groups[specimen_label] = {}

    groups[specimen_label][view_num] = os.path.join(photodir, fname)

if len(groups) == 0:
    raise RuntimeError(
        "No view images were found.\n"
        "Expected filenames like specimen_view1.png ... specimen_view4.png"
    )

specimen_names = sorted(groups.keys())

# ============================================================
# Build GIFs
# ============================================================

gif_records = []

duration_ms = 700

for specimen_label in specimen_names:
    image_paths = groups[specimen_label]

    ordered_paths = [
        image_paths[view_num]
        for view_num in sorted(image_paths.keys())
    ]

    frames = [
        Image.open(path).convert("RGB")
        for path in ordered_paths
    ]

    outfile = os.path.join(
        gif_outdir,
        f"{specimen_label}.gif"
    )

    frames[0].save(
        outfile,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0
    )

    gif_records.append({
        "specimen_label": specimen_label,
        "n_frames": len(frames),
        "gif_path": outfile,
    })

    print(f"Saved GIF: {outfile}")

# ============================================================
# Write GIF log
# ============================================================

gif_log_csv = os.path.join(
    gif_outdir,
    f"optional_gifs_log_{md.dataset_folder_name}_{timestamp}.csv"
)

pd.DataFrame(gif_records).to_csv(
    gif_log_csv,
    index=False
)

# ============================================================
# Confirm completion
# ============================================================

print_success("GIF generation complete.")

print_step_complete_header("Optional GIF building complete")

print("GIF folder:")
print()
print(f"    {gif_outdir}")
print()

print("GIF log:")
print()
print(f"    {gif_log_csv}")
print()

print(f"Specimens included: {len(specimen_names)}")
print()
