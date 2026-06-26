#!/usr/bin/env python3

"""
05_clean_meshes.py

Original processing logic: RileyWilde
Refactoring and workflow design: Katrina E. Yezzi-Woodley

Loads cleaned surface meshes (.ply) from the current scan's
Clean_Meshes folder, aligns each mesh using principal component analysis (PCA),
and renders standardized preview images from four viewpoints.

Outputs are written to:
    <scanpath>/scanphotos
"""

from utils import *

try:
    import pyvista as pv
except ImportError:
    raise RuntimeError(
        "PyVista is required to render publication images.\n"
        "Install with: pip install pyvista\n"
        "Or skip this step if you do not need image rendering."
    )
    
try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    raise RuntimeError(
        "Pillow is required to render publication images.\n"
        "Install with: pip install Pillow\n"
        "Or skip this step if you do not need image rendering."
    )

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
meshdir = os.path.normpath(os.path.join(scanpath, "Clean_Meshes"))
outdir = os.path.normpath(os.path.join(scanpath, "scanphotos"))

if not os.path.isdir(meshdir):
    raise RuntimeError(
        f"Clean mesh folder not found: {meshdir}\n"
        "Run 05_clean_meshes.py first, or check your scanpath."
    )

os.makedirs(outdir, exist_ok=True)

# Collect only .ply meshes
mesh_paths = [
    os.path.join(meshdir, f)
    for f in os.listdir(meshdir)
    if f.lower().endswith(".ply")
]

if len(mesh_paths) == 0:
    raise RuntimeError(
        f"No .ply files found in {meshdir}.\n"
        "Run 05_clean_meshes.py first or confirm that cleaned meshes were created."
    )

print(f"Found {len(mesh_paths)} cleaned mesh(es) in {meshdir}")
print(f"Images will be written to {outdir}")


def choose_scale_bar_length_mm(world_width_mm):
    """
    Choose a nice round scale-bar length that fits comfortably
    within the rendered field width.
    """
    candidates = [0.5, 1, 2, 5, 10, 20, 50]
    max_allowed = 0.25 * world_width_mm  # bar should occupy at most ~25% of width

    valid = [x for x in candidates if x <= max_allowed]
    if len(valid) == 0:
        return candidates[0]
    return valid[-1]


def add_scale_bar_to_png(
    png_path,
    bar_length_mm,
    world_width_mm,
    margin_px=60
):
    """
    Draw a scale bar onto an existing PNG.

    world_width_mm = total real-world width visible across the rendered image
    """
    img = Image.open(png_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    
    W, H = img.size

    # Scale relative to image size
    font_size = int(W * 0.035)   # ~3.5% of image width
    bar_height_px = int(W * 0.008)

    px_per_mm = W / world_width_mm
    bar_length_px = int(round(bar_length_mm * px_per_mm))

    x0 = margin_px
    y0 = H - margin_px
    x1 = x0 + bar_length_px
    y1 = y0 - bar_height_px

    # black bar
    draw.rectangle([x0, y1, x1, y0], fill="white")

    label = f"{bar_length_mm:g} mm"

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    draw.text(
        (x0, y1 - font_size - 10),
        label,
        fill="white",
        font=font,
        stroke_width=2,
        stroke_fill="black"
    )

    img.save(png_path)

def mesh_images(mesh_path, outdir):
    """
    Load one mesh, align it by PCA, and save 4 standard views.
    """
    try:
        print(f"Rendering: {mesh_path}")

        pts, tri = tm.read_ply(mesh_path)

        stem = os.path.splitext(os.path.basename(mesh_path))[0]

        # Center mesh at origin
        pts = pts - pts.mean(0)

        # PCA-style alignment from covariance-like matrix
        vals, vecs = scipy.linalg.eig(pts.T @ pts)
        ind = np.isreal(vals)
        vals = np.real(vals[ind])
        vecs = np.real(vecs[:, ind])

        # Reorder axes so:
        # x = middle variance axis
        # y = smallest variance axis
        # z = largest variance axis
        a = np.arange(3)
        reorder = np.array([
            a[(a != vals.argmax()) * (a != vals.argmin())].item(),
            vals.argmin(),
            vals.argmax()
        ])

        vecs = vecs[:, reorder]

        # Force right-handed orientation
        if np.linalg.det(vecs) < 0:
            vecs[:, -1] = -1 * vecs[:, -1]

        p2 = pts @ vecs
        p2 = p2 - p2.min(0)
        
        # Heuristic flip: put broader mass toward the top of the long axis
        z = p2[:, 2]
        zmin = z.min()
        zmax = z.max()
        zspan = zmax - zmin

        if zspan > 0:
            bottom_mask = z < (zmin + 0.20 * zspan)
            top_mask = z > (zmax - 0.20 * zspan)

            bottom_width = np.ptp(p2[bottom_mask, 0]) + np.ptp(p2[bottom_mask, 1])
            top_width = np.ptp(p2[top_mask, 0]) + np.ptp(p2[top_mask, 1])

            if bottom_width > top_width:
                p2[:, 2] = zmax - p2[:, 2]

        # Specimen color
        cb = (201 / 255, 167 / 255, 126 / 255)

        faces = np.hstack(
            [np.full((tri.shape[0], 1), 3, dtype=np.int64), tri.astype(np.int64)]
        )

        mesh = pv.PolyData(p2, faces)

        plotter = pv.Plotter(off_screen=True, window_size=(1800, 1800))
        plotter.set_background("black")
        plotter.add_mesh(mesh, color=cb)
        plotter.camera.parallel_projection = True

        xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
        cx, cy, cz = mesh.center

        xspan = xmax - xmin
        yspan = ymax - ymin
        zspan = zmax - zmin
        d = 3 * max(xspan, yspan, zspan)

        views = {
            "view1": [(cx + d, cy, cz), (cx, cy, cz), (0, 0, 1)],
            "view2": [(cx, cy + d, cz), (cx, cy, cz), (0, 0, 1)],
            "view3": [(cx - d, cy, cz), (cx, cy, cz), (0, 0, 1)],
            "view4": [(cx, cy, cz + d), (cx, cy, cz), (0, 1, 0)],
        }

        for label, cpos in views.items():
            plotter.camera_position = cpos
            plotter.camera_set = True
            plotter.render()
                        
            world_width_mm = max(xspan, yspan)
                
            outpng = os.path.join(outdir, stem + f"_{label}.png")
            plotter.screenshot(outpng)

            bar_length_mm = choose_scale_bar_length_mm(world_width_mm)
            add_scale_bar_to_png(
                png_path=outpng,
                bar_length_mm=bar_length_mm,
                world_width_mm=world_width_mm
            )

        plotter.close()

    except Exception as error:
        print(f"Rendering error with {mesh_path}: {error}")


num_cores = max(1, min(multiprocessing.cpu_count(), len(mesh_paths)))

Parallel(n_jobs=num_cores)(
    delayed(mesh_images)(mesh_path, outdir) for mesh_path in mesh_paths
)

print("Image render finished.")
