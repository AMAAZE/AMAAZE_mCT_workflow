"""
03_segment.py

Original processing logic: RileyWilde
Refactoring and workflow design: Katrina E. Yezzi-Woodley

Load the processed volume and segment it into tiers and specimen regions
using manual divider definition informed by the scan layout.

This step:
- determines vertical tier boundaries,
- allows the user to define row/column dividers per tier,
- computes specimen extraction boxes,
- and saves the extraction plan to CT<scan_num>.csv
for downstream extraction and surfacing.

"""

# ============================================================
# Validate required input paths
# ============================================================

from utils import *

slicepath = os.path.normpath(slicepath)

if not os.path.isdir(slicepath):
    raise RuntimeError("The provided slicepath was not found.")

if not os.path.exists(layoutfile): 
    raise RuntimeError("The layoutfile was not found.")

# ============================================================
# Legacy segmentation parameter file
# ============================================================
# REVIEW / LIKELY LEGACY.
#
# This creates a text file like ct2_params.txt inside the scan folder.
# That file stores segmentation settings and remembered user choices.
#
# Why it probably existed:
# - It lets the workflow remember values between runs.
# - It avoids asking the user to redo every segmentation decision.
#
# Why it is now a problem:
# - It creates hidden state outside user_inputs.json.
# - The same JSON file can behave differently if params.txt already exists.
# - It is not very readable or structured.
#
# Current recommendation:
# DO NOT DELETE YET.
# First identify exactly which values are still needed.
# Later, replace this with something explicit like:
#   ct<scan_num>_segmentation_metadata.json
# containing tier boundaries, thresholds, divider positions, rotations, etc.

# This block creates and reads ct<scan_num>_params.txt.
#
# The file stores segmentation-related settings and user choices
# between runs, including rotation, divider detection behavior,
# tier thresholds, and manual overrides.
#
# NOTE(dev):
# This system creates hidden state outside user_inputs.json.
# It is currently retained for compatibility with the existing
# workflow and downstream scripts.
#
# Future refactor goal:
# Replace with an explicit structured metadata file.

params_fname = os.path.join(scanpath, f"ct{scan_num}_params.txt")
if not os.path.exists(params_fname):
    with open(params_fname, 'w') as f:
        f.write("AUTO_ROT= True\n")
        f.write("AUTO_SEG= True\n")
        f.write("INPUT_ROWS= False\n")
        f.write("INPUT_COLS= False\n")
        f.write("TIER_THRESH= None\n")
        f.write("INVERT= False\n")
        f.write("THRESH0= 500000.0\n")
        f.write("THRESH1= 500000.0\n")

# REVIEW.
# This reads the legacy params file back into Python variables.
# These variables control major behavior later in the script.
#
# AUTO_ROT    = automatically estimate divider-grid rotation per tier.
# AUTO_SEG    = automatically detect divider positions from row/column signals.
# INPUT_ROWS  = old manual row-divider path.
# INPUT_COLS  = old manual column-divider path.
# TIER_THRESH = saved vertical tier boundary positions.
# INVERT      = invert intensity values.
# THRESH0/1   = older manual threshold values for divider detection.
    
AUTO_ROT, AUTO_SEG, INPUT_ROWS, INPUT_COLS, TIER_THRESH, INVERT, THRESH0, THRESH1 = read_hyperparameters(
    scan_num, directory=scanpath
)

# REVIEW.
# These force the script to redo tier and threshold choices every run.
# This is useful while debugging, but probably should become config-controlled.
#
# REDO_TIERS=True means it will ask for tier boundary confirmation each time.
# REDO_ISO=True means it will redo T1/T2/T3 threshold selection each time.
# PLOTTING=True means it will show diagnostic plots.

REDO_TIERS=True
PLOTTING = True

# ============================================================
# Load TIFF filenames and reduced working volume
# ============================================================
# KEEP.
# The TIFF filenames are needed later for display and for mapping back to slice indices.
# The reduced .npz volume is the main object used for tier and divider/cell segmentation.

fnames = []
for ext in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
    fnames.extend(glob.glob(os.path.join(slicepath, ext)))

fnames = sorted(set(fnames))
fnames_with_idx = [(f, extract_index(f)) for f in fnames]
fnames_with_idx.sort(key=lambda x: x[1])

fnames = [f for f, _ in fnames_with_idx]
indices = [idx for _, idx in fnames_with_idx]

LOCAL_SLICES = len(fnames) > 0

npz_fname = os.path.join(scanpath, f"ct{scan_num}_new.npz")
if not os.path.exists(npz_fname):
    raise RuntimeError(f"Processed volume not found: {npz_fname}. Run 01_set_rotation_crop.py first.")

saveddata = np.load(npz_fname)
vol = saveddata["vol"]
rowrng = saveddata["rowrng"]
colrng = saveddata["colrng"]
ang2rot = saveddata["ang"]
origsz = saveddata["origsz"]
rem = saveddata["remainder"]
transpose_preview = bool(saveddata["transpose_preview"])

# KEEP.
# These are original cropped TIFF dimensions.
# They are used later to map reduced-volume coordinates back to original TIFF coordinates.
#
# Important issue:
# If 02_build_volume.py forces the reduced volume to 225 x 225, then vol.shape[1]
# and vol.shape[2] are artificial working dimensions, not physical/native dimensions.

rowsz = rowrng[1]-rowrng[0]
colsz = colrng[1]-colrng[0]

# REVIEW.
# This flips intensity logic if needed.
# Keep for now, but document when it is actually needed.

if INVERT:
    vol = vol.max() - vol

# ============================================================
# 3. Read layout file
# ============================================================
# KEEP.
#
# The layout CSV tells the workflow what specimens are supposed to be
# in each cell of the packaging grid.
#
# This does NOT detect dividers. It only tells the workflow:
# - how many tiers are expected
# - how many rows/columns are expected
# - what each extracted cell should be named
#
# Example idea:
# tier 1, row 1, col 1 = specimen_A
# tier 1, row 1, col 2 = specimen_B
#
# Empty cells are converted to 0 by fillna(0), and later ignored.

x = pd.read_csv(layoutfile).fillna(0)
x = x.to_numpy()

# Keep only the rows for the scan currently being processed.
# Column 0 is scan number, so after filtering we remove it with [:, 1:].
x = x[x[:,0]==scan_num,1:].copy()
    
if len(x) == 0:
    raise ValueError(
        f'Scan {scan_num} not found in layout file. '
        'Check that the CSV includes this scan and follows the required format.'
    )

# Number of tiers listed in the layout file.
n_tiers = int(x[:,0].max())

# Number of rows listed in the layout file.
dim1 = int(x[:,1].max())

# Number of specimen columns.
# The first two columns are tier and row, so the rest are specimen cells.
dim2 = x.shape[1]-2

# Build a 3D layout array:
# scan_layout[tier, row, column]
#
# This is not image data. It is the specimen name/label map.
scan_layout = np.zeros((n_tiers,dim1,dim2),object)
for i in range(n_tiers):
    scan_layout[i,:,:] = x[x[:,0]==i+1,2:]

# mask marks which cells actually contain specimens.
# Empty cells are ignored during extraction.
mask = scan_layout!=0 #cells to extract from
tier_mask = np.sum(np.sum(mask,1),1)>0

# ============================================================
# Segment tiers in Z
# ============================================================
# KEEP, BUT REVIEW INTERFACE.
#
# This step works from the reduced .npz volume, not directly from the
# original TIFF stack.
#
# It collapses each Z slice into one average value, producing a 1D signal
# through the stack. Peaks/dips in that signal are used to identify tier
# boundaries.
#
# This part seems conceptually useful and the graph is readable.
# The main thing to review later is how accepted tier boundaries are saved.

q = -np.mean(np.mean(vol,1),1)
sig = q.copy()
q = q - q.min()

thresh = 3e7/(vol.shape[1]*vol.shape[2])
q[q<thresh] = 0

# NOTE: peak width controls minimum feature size detected in the 1D projection.
# If divider contrast is low or features are narrow, this may need adjustment.
# This is a heuristic parameter, not a fixed physical constant.
vert_pks = find_peaks(q,width=10)[0]
vert_pks = np.concatenate((np.array([0]),vert_pks))
vert_pks = np.concatenate((vert_pks,np.array([len(q)])) )

fig = plt.figure()
plt.plot(np.arange(len(q)),sig)
plt.xlabel('slice height (z)'); plt.ylabel('(-) average tier density'); plt.title('tier segmentation')
for vvv in vert_pks:
    plt.axvline(x=vvv, color='green', linestyle='--', linewidth=2)

if (TIER_THRESH is None) or (REDO_TIERS is True):
    print(
        "green lines are candidate vertical tier-boundary peaks. \n"
        "Click the final tier boundary positions if the candidates are not correct. \n"
        "# tiers is %1d, # nonempty tiers is %2d \n" % (n_tiers, np.sum(tier_mask))
    )
        
    clicked_x = []  # store clicked x-values

    def onclick(event):
        if event.inaxes:
            x_click = event.xdata
            clicked_x.append(x_click)
            # Draw vertical line
            event.inaxes.axvline(x_click, color="r", linestyle="--")
            plt.draw()
            print(f"Clicked x = {x_click:.2f}")

    cid = fig.canvas.mpl_connect("button_press_event", onclick)

    plt.show(block=False)        
        
    a = input("please press enter once done (no clicks = use suggested values) \n")
    yn_vertseg = clicked_x #input('are these ok? enter y/n. # tiers is %1d, # nonempty tiers is %2d \n' % (n_tiers, np.sum(tier_mask))#)

    if len(yn_vertseg) > 0:
        ex = np.array(yn_vertseg).astype(int)
        ex[ex<0] = 0
        ex[ex>len(q)] = len(q)
        print("new vertical peaks", ex)
    else:
        ex = vert_pks
        print("using ", ex)
    update_param(scan_num, 'TIER_THRESH', ex.tolist(), directory=scanpath)
else:
    print("using saved vertical peaks")
    ex = np.array(TIER_THRESH)

# Convert the selected tier boundary positions into start/end ranges.
# Each range is one tier in the reduced .npz volume.
#
# REVIEW:
# The ranges are reversed here. This may be intentional because of scan/layout
# orientation, but we should confirm before changing it.

ranges = []
for i in range(len(ex)-1):
    ranges.append([ex[i],ex[i+1]])

ranges = ranges[-1::-1]
tiers = np.arange(n_tiers)[tier_mask]
ranges = [ranges[i] for i in tiers]

# Pull out the reduced-volume data for each tier.
# Each item in SLICES is one tier-sized chunk of the .npz volume.
#
# This means the downstream divider/cell segmentation is operating on
# reduced .npz data, not directly on the full TIFF stack.

SLICES = []
for i in range(len(ranges)):
    SLICES.append(vol[ranges[i][0]:ranges[i][1],:,:])

I = [x.mean(0) for x in SLICES]

# This will store the final extraction instructions.
# At the end of the script, these become CT<scan_num>.csv.
#
# That CSV is used by 04_surface.py to go back to the original TIFF stack and
# extract each specimen.

EXTRACTS = []

# ============================================================
# Segment dividers/cells within each tier
# ============================================================
# REVIEW.
#
# This is the most tangled part of the script.
#
# For each tier, the workflow:
# 1. uses T1/T2/T3 to create an image-like divider signal,
# 2. tries to identify row and column divider positions,
# 3. turns those divider positions into cell boxes,
# 4. maps those boxes back from reduced .npz space into original TIFF space,
# 5. saves those extraction boxes for 04_surface.py.
#
# This is where several issues overlap:
# - graph-based divider detection works on some datasets but not others,
# - click/manual approaches work on some datasets but not others,
# - the reduced .npz volume may not preserve aspect ratio,
# - divider positions may differ by tier,
# - and the current logic is hard to interpret.

for i in range(len(tiers)):
    tier = tiers[i]
    Im = I[i]
    angi = 0

    maski = mask[i,:,:]
    layouti = scan_layout[i,:,:]

    rowi, coli = np.where(maski)

    accepted_grid = False

    while not accepted_grid:

        print(f"\nTier {i+1}: manual grid definition")

        print(
            "Click once on each horizontal row divider.\n"
            "Press Enter in the terminal when finished."
        )

        i1m = collect_divider_line_clicks_free(
            Im,
            axis_label="row",
            title=f"Tier {i+1}: click row dividers"
        )

        print(
            "Click once on each vertical column divider.\n"
            "Press Enter in the terminal when finished."
        )

        i0m = collect_divider_line_clicks_free(
            Im,
            axis_label="col",
            title=f"Tier {i+1}: click column dividers"
        )

        print(f"Tier {i+1} row divider coordinates: {i1m}")
        print(f"Tier {i+1} column divider coordinates: {i0m}")

        plt.figure()
        plt.title(f"Tier {i+1}: proposed manual grid")
        plt.imshow(Im, cmap="gray")

        for rr in i1m:
            plt.axhline(rr, color='red', linestyle='--')

        for cc in i0m:
            plt.axvline(cc, color='blue', linestyle='--')

        plt.xlabel("X coordinate")
        plt.ylabel("Y coordinate")

        plt.show(block=False)

        accept_grid = input(
            f"Accept manual grid for tier {i+1}? (y/n): "
        ).strip().lower()

        accepted_grid = (accept_grid == "y")

    print(f"Tier {i+1} accepted row dividers: {i1m}")
    print(f"Tier {i+1} accepted column dividers: {i0m}")  

    colstart = 0
    colend = Im.shape[1]-1
    rowstart = 0
    rowend = Im.shape[0]-1

    col = np.array( [colstart]+ i0m.tolist() +[colend])
    row = np.array( [rowstart]+ i1m.tolist() +[rowend])

    # Convert divider positions into cell boxes.
    #
    # row and col contain the boundaries of each cell.
    # rowi and coli come from the layout mask and identify which cells contain
    # specimens.
    #
    # rowcolrng stores:
    # [row_start, row_end, col_start, col_end]
    # for each specimen cell.
    
    """
    ============================================================
    DEBUG: Extraction Box Construction
    ============================================================
    Purpose:
    Verify divider arrays and layout indexing before
    row/column extraction boxes are generated.
    ============================================================
    """
    
    print(f"Tier {i+1} extraction debug:")
    print(f"  row array length: {len(row)}")
    print(f"  col array length: {len(col)}")
    print(f"  rowi max: {rowi.max()}")
    print(f"  coli max: {coli.max()}")
    print(f"  row values: {row}")
    print(f"  col values: {col}")
    
    """
    ============================================================
    END DEBUG: Extraction Box Construction
    ============================================================
    """

    if rowi.max()+1 >= len(row):
        raise RuntimeError(
            f"Tier {i+1}: not enough row dividers were clicked "
            f"for the occupied layout cells."
        )

    if coli.max()+1 >= len(col):
        raise RuntimeError(
            f"Tier {i+1}: not enough column dividers were clicked "
            f"for the occupied layout cells."
        )

    rowcolrng = np.vstack((row[rowi],row[rowi+1],col[coli],col[coli+1])).T
    namesi = layouti[maski,None]

    # Choose a representative original TIFF slice for plotting the segmentation
    # boxes back on top of the real image.
    #
    # This is for visual checking only.
    # It does not create the extraction boxes.

    slice_idx = int(np.mean(ranges[i])) * zwindow
        
    if PLOTTING and LOCAL_SLICES:
        slice_idx = max(0, min(slice_idx, len(fnames) - 1))
                    
        # IMPORTANT: fnames must remain full paths (used directly in plt.imread)
        imdisp = apply_preview_orientation(
            plt.imread(fnames[slice_idx]),
            transpose_preview
        )
        imdisp = rotate(imdisp, ang2rot, preserve_range=True, resize=True)
        imdisp = imdisp[rowrng[0]:rowrng[1], colrng[0]:colrng[1]].copy()
        imdisp = rotate(imdisp, angi)

    # Map cell boxes from reduced .npz coordinates back into cropped TIFF coordinates.
    #
    # This is a critical coordinate transform.
    #
    # Current assumption:
    #   reduced row coordinates scale by rowsz / vol.shape[1]
    #   reduced col coordinates scale by colsz / vol.shape[2]
    #
    # REVIEW:
    # This is where square resizing / aspect-ratio changes matter.
    # If 02_build_volume.py forces 225 x 225, this transform can still work
    # numerically, but the reduced segmentation representation may distort
    # the grid before boxes are mapped back.

    drawrow = ((rowsz / vol.shape[1]) * (rowcolrng[:,0:2])).astype(int)
    drawcol = ((colsz / vol.shape[2]) * (rowcolrng[:,2:4])).astype(int)

    # Store extraction instructions for this tier.
    #
    # These instructions include:
    # - specimen name,
    # - Z range in original slice indices,
    # - row/column crop box in original TIFF space,
    # - tier-specific rotation,
    # - original crop settings.
    #
    # 04_surface.py will use this CSV to extract specimen subvolumes from the
    # original TIFF stack.

    EXTRACTS.append(
        np.concatenate(
            (
                namesi,
                [[zwindow * ranges[i][0], zwindow * ranges[i][1]]] * rowcolrng.shape[0], 
                drawrow, 
                drawcol, 
                len(namesi)*[[angi]], 
                len(namesi)*[rowrng.tolist()], 
                len(namesi)*[colrng.tolist()] 
                ),
                1
            )
    )

#     if LOCAL_SLICES and PLOTTING :
#        draw_boxes(imdisp,drawrow,drawcol,title = 'segmentation for tier '+str(i+1))

E = np.concatenate(EXTRACTS)

# ============================================================
# 7. Write extraction plan
# ============================================================
# KEEP.
#
# E contains the extraction plan for all specimens in all tiers.
#
# This CSV is the handoff from segmentation to surfacing.
# 04_surface.py reads it to extract each specimen from the original TIFF stack.
#
# REVIEW:
# The CSV works, but a future structured format may be easier to debug.
# For now, keep it because downstream code expects it.

pd.DataFrame(E).to_csv(
    os.path.join(scanpath, f"CT{scan_num}.csv"),
    header=False,
    index=False
)

