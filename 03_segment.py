#!/usr/bin/env python3

"""
03_segment.py

Original processing logic: RileyWilde
Refactoring and workflow design: Katrina E. Yezzi-Woodley

Load the processed volume and segment it into tiers and individual
specimen regions using intensity-based thresholds and scan layout information.

This step determines vertical tier boundaries, identifies divider structure,
and computes bounding boxes for each specimen, saving the extraction plan
to CT<scan_num>.csv for downstream volume extraction and surfacing.
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
REDO_ISO = True
PLOTTING = True

# ============================================================
# Load TIFF filenames and reduced working volume
# ============================================================
# KEEP.
# The TIFF filenames are needed later for display and for mapping back to slice indices.
# The reduced .npz volume is the main object used for tier and divider/cell segmentation.

fnames = glob.glob(os.path.join(slicepath, "*.tif"))

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
        "green are identified vertical peaks - are they ok? \n"
        "if not, please click new peaks. \n"
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

# Show the voxel/intensity distribution of the reduced .npz volume.
# This helps the user understand what intensity values exist before choosing
# thresholds.

fig = plt.figure(figsize=(10,4))
plt.hist(vol.flatten(), bins=500)
plt.title("voxel value histogram")
plt.ylabel("frequency")
plt.xlabel("voxel value")

# ============================================================
# Choose intensity thresholds for divider/specimen detection
# ============================================================
# REVIEW.
#
# This chooses T1, T2, and T3.
#
# These are not voxel sizes. They are intensity thresholds.
#
# They are used later by id_cardboard() or autorot2() to decide which parts
# of the tier volume look like divider material and which parts look like
# specimen material.
#
# IMPORTANT CURRENT ISSUE:
# The click-based threshold method currently samples from the original TIFF
# images, while the divider/cell segmentation later uses the reduced .npz
# volume. That means thresholds are selected in one representation and applied
# in another.
#
# FIRST EXPERIMENTAL QUESTION:
# Would threshold selection behave more consistently if the user clicked on
# a representative image generated from the reduced .npz volume instead?

iso_override = iso_thresholds_override

if iso_override is not None:
    divider_range = np.array(iso_override).astype(int)
    print(f"Using ISO threshold override from JSON: {divider_range}")

# elif REDO_ISO:
#    t1t2t3 = get_iso_thresholds_from_voxel_probe(
#        slicepath=slicepath,
#        slice_index_fraction=slice_index_fraction,
#        transpose_preview=transpose_preview,
#        ang2rot=ang2rot,
#        rowrng=rowrng,
#        colrng=colrng,
#        n_clicks=voxel_probe_n_clicks
#    )

elif REDO_ISO:
    # Experimental change:
    # Select T1/T2/T3 from the reduced .npz segmentation volume rather than
    # from the original TIFF stack.
    #
    # This keeps threshold selection in the same data representation used by
    # divider/cell segmentation below.

    slice_index = int(len(fnames) * slice_index_fraction)
    slice_index = min(slice_index, len(fnames) - 1)

    npz_slice_index = slice_index // zwindow
    npz_slice_index = max(0, min(npz_slice_index, vol.shape[0] - 1))

    print(f"Using reduced .npz slice {npz_slice_index} for threshold selection.")
    print(f"This corresponds approximately to TIFF slice index {slice_index}.")

    probe_image = vol[npz_slice_index, :, :]


    # ============================================================
    # DEBUG: Threshold Probe Mapping
    # ============================================================
    # Purpose:
    # Verify that TIFF-space slice selection is mapping correctly
    # into reduced .npz space before threshold clicking.
    #
    # Remove or reduce once stable.
    # ============================================================

    print("Threshold probe debug:")
    print(f"  TIFF slice count: {len(fnames)}")
    print(f"  slice_index_fraction: {slice_index_fraction}")
    print(f"  selected TIFF slice index: {slice_index}")
    print(f"  zwindow: {zwindow}")
    print(f"  reduced .npz volume shape: {vol.shape}")
    print(f"  selected .npz slice index: {npz_slice_index}")
    print(f"  probe image shape: {probe_image.shape}")
    print(f"  probe image min/max: {probe_image.min()} / {probe_image.max()}")

    # ============================================================
    # END DEBUG: Threshold Probe Mapping
    # ============================================================

    divider_range = get_divider_range_from_image_probe(
        probe_image=probe_image,
        n_clicks=voxel_probe_n_clicks
    )

    print(f"Final threshold values used for segmentation: {divider_range}")

    update_param(scan_num, "ISO_THRESHOLDS", divider_range.tolist(), directory=scanpath)

else:
    divider_range = get_parameter(scan_num, "ISO_THRESHOLDS", directory=scanpath)

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
    tier = tiers[i]; # Im = I[i].T
    si = SLICES[i]

    # Create a divider-detection image for this tier.
    #
    # si is the reduced .npz data for this tier.
    #
    # AUTO_ROT=True:
    #   autorot2() tries to estimate rotation of the divider grid and create
    #   the divider signal.
    #
    # AUTO_ROT=False:
    #   id_cardboard() creates the divider signal without estimating rotation.
    #
    # REVIEW:
    # The names id_cardboard() and autorot2() are not very transparent.
    # # Also, both depend heavily on whether divider_range correctly captures
    # the divider material intensity range.

#    if AUTO_ROT:
#        ''' same params go into id_cardboard:'''
#        angi,Im = autorot2(
#            si,
#            t1=t1t2t3[0],
#            t2=t1t2t3[1],
#            t3=t1t2t3[2],
#            title="rotation for tier "+str(i+1)
#        )
#
#    else:
#        ''' same params go into autorot2:'''
#        Im = id_cardboard(
#            si,
#            t1=t1t2t3[0],
#            t2=t1t2t3[1],
#            t3=t1t2t3[2]
#        )
#        angi = 0

    if AUTO_ROT:
        ''' same params go into id_cardboard:'''
        angi, Im = autorot2(
            si,
            divider_low=divider_range[0],
            divider_high=divider_range[1],
            title="rotation for tier " + str(i+1)
        )

    else:
        ''' same params go into autorot2:'''
        Im = id_cardboard(
            si,
            divider_low=divider_range[0],
            divider_high=divider_range[1]
        )
        angi = 0

    plt.figure()
    plt.title("detected dividers for tier "+str(i+1))
    plt.imshow(Im)
    plt.axis("off")
    plt.show(block=False)

    angi = -angi 

    maski = mask[i,:,:]
    layouti = scan_layout[i,:,:]

    rowi,coli = np.where(maski)

    Im2 = Im.copy()
    
    # ============================================================
    # Tier-specific expected divider counts
    # ============================================================

    expected_row_dividers = maski.shape[0] - 1
    expected_col_dividers = maski.shape[1] - 1

    print(f"Tier {i+1} expected row dividers from layout: {expected_row_dividers}")
    print(f"Tier {i+1} expected col dividers from layout: {expected_col_dividers}")
    
    # Collapse the divider image into row/column signals.
    #
    # In plain language:
    # - Isum0 helps find column dividers.
    # - Isum1 helps find row dividers.
    #
    # This is the graph-based part of divider/cell segmentation.
    Isum0 = np.sum(Im,0)
    Isum1 = np.sum(Im,1)

    # Decide how row/column divider positions will be found.
    #
    # Possible paths:
    # 1. JSON overrides exist:
    #    Use user-provided divider positions.
    #
    # 2. AUTO_SEG=True:
    #    Try to find divider positions automatically from the row/column signals.
    #
    # 3. AUTO_SEG=False:
    #    Use older threshold logic based on THRESH0 and THRESH1.
    #
    # REVIEW:
    # This is one of the places where old logic, new overrides, and automation
    # are mixed together.
    
    if (row_dividers_override is not None) or (col_dividers_override is not None):
        x0 = np.zeros_like(Isum0, dtype=int)
        x1 = np.zeros_like(Isum1, dtype=int)
    elif AUTO_SEG == True: # Automatic segmentation
        x0 = auto_seg(Isum0, expected_col_dividers)
        x1 = auto_seg(Isum1, expected_row_dividers)
    else:
        thresh0 = Im.shape[0] * THRESH0 / 225
        thresh1 = Im.shape[1] * THRESH1 / 225

        x0 = (Isum0 > thresh0).astype(int)
        x1 = (Isum1 > thresh1).astype(int)

    if x0[0]==1:
        x0[0]=0

    i0firsts = np.where(x0[0:-1]<x0[1:])[0]
    i0lasts  = np.where(x0[1:]<x0[0:-1])[0]
    i1firsts = np.where(x1[0:-1]<x1[1:])[0]
    i1lasts  = np.where(x1[1:]<x1[0:-1])[0]

    # Optional manual row-divider path.
    #
    # This is legacy behavior controlled by params.txt.
    # It shows diagnostic plots and asks the user to enter row-divider positions.
    #
    # REVIEW:
    # This is separate from the newer image-click threshold selection.
    # The name INPUT_ROWS is not very descriptive.
    if INPUT_ROWS: # Click selection manual override
        plt.figure()
        plt.imshow(Im)

        plt.figure()
        plt.plot(Isum1,scalex=5)
        plt.xticks(np.arange(0,Isum1.shape[0],20))
        plt.show(block=False)
        print(find_peaks(Isum1,width=3)[0])
        i1m = np.array(input('enter peaks, separated only by single spaces: \n').split(' ')).astype(int)
        update_param(scan_num,'R'+str(i),i1m, directory=scanpath)
    elif row_dividers_override is not None:
        i1m = np.round(row_dividers_override).astype(int)
    else:
        i1m = np.floor((i1firsts+i1lasts)/2).astype(int)

    # Optional manual column-divider path.
    #
    # Same idea as INPUT_ROWS, but for column dividers.
    #
    # REVIEW:
    # This is legacy behavior and should probably be renamed or replaced later.
    
    if INPUT_COLS:
        plt.figure()
        plt.plot(Isum0,scalex=5)
        plt.xticks(np.arange(0,Isum1.shape[0],20))
        plt.show(block=False)
        print(find_peaks(Isum0,width=3)[0])
        i0m = np.array(input('enter peaks, separated only by single spaces: \n').split(' ')).astype(int)
        update_param(scan_num,'C'+str(i),i0m, directory=scanpath)
    elif col_dividers_override is not None:
        i0m = np.round(col_dividers_override).astype(int)
    else:
        i0m = np.floor((i0firsts+i0lasts)/2).astype(int)

    print("tier", i+1, "final i1m (row dividers):", i1m)
    print("tier", i+1, "final i0m (col dividers):", i0m)
    
    """
    ============================================================
    DEBUG: Tier Divider Proposal
    ============================================================
    Purpose:
    Verify automatic divider detection before user acceptance.
    Remove or reduce once stable.
    ============================================================
    """

    print(f"Tier {i+1} segmentation debug:")
    print(f"  Im shape: {Im.shape}")
    print(f"  Proposed row dividers i1m: {i1m}")
    print(f"  Proposed col dividers i0m: {i0m}")
    print(f"  Expected row dividers: {expected_row_dividers}")
    print(f"  Expected col dividers: {expected_col_dividers}")
    print(f"  AUTO_SEG: {AUTO_SEG}")
    print(f"  AUTO_ROT: {AUTO_ROT}")

    """
    ============================================================
    END DEBUG: Tier Divider Proposal
    ============================================================
    """
    
    # ============================================================
    # Per-tier segmentation verification
    # ============================================================

    plt.figure()
    plt.title(f"Proposed divider segmentation for tier {i+1}")
    plt.imshow(Im)
    
    for rr in i1m:
        plt.axhline(rr, color='red', linestyle='--')

    for cc in i0m:
        plt.axvline(cc, color='blue', linestyle='--')

    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")

    plt.show(block=False)

    accept_seg = input(
        f"Accept proposed divider segmentation for tier {i+1}? (y/n): "
    ).strip().lower()

    if accept_seg != "y":

        print(f"Manual divider selection for tier {i+1}")

        i1m = collect_divider_line_clicks(
            Im,
            n_lines=expected_row_dividers,
            axis_label="row"
        )

        i0m = collect_divider_line_clicks(
            Im,
            n_lines=expected_col_dividers,
            axis_label="col"
        )

        print("Manual row dividers:", i1m)
        print("Manual col dividers:", i0m)
        
        """
        ============================================================
        DEBUG: Manual Divider Correction
        ============================================================
        Purpose:
        Verify user-selected divider coordinates after manual override.
        Remove or reduce once stable.
        ============================================================
        """

        print(f"Tier {i+1} manual correction debug:")
        print(f"  Manual row dividers i1m: {i1m}")
        print(f"  Manual col dividers i0m: {i0m}")

        """
        ============================================================
        END DEBUG: Manual Divider Correction
        ============================================================
        """

        # Show corrected divider overlay
        plt.figure()
        plt.title(f"Corrected divider segmentation for tier {i+1}")
        plt.imshow(Im)

        for rr in i1m:
            plt.axhline(rr, color='red', linestyle='--')

        for cc in i0m:
            plt.axvline(cc, color='blue', linestyle='--')

        plt.xlabel("X coordinate")
        plt.ylabel("Y coordinate")

        plt.show(block=False)

    # Show diagnostic row/column signal plots.
    # These help the user see where the workflow thinks dividers are.
    #
    # REVIEW:
    # These plots are useful, but the labels may be confusing.
    # Also, their usefulness varies by dataset.
    
    """
    ============================================================
    DEBUG: Final Divider Coordinates
    ============================================================
    Purpose:
    Confirm final divider coordinates before extraction box generation.
    Remove or reduce once stable.
    ============================================================
    """

    print(f"Tier {i+1} final divider coordinates:")
    print(f"  Rows: {i1m}")
    print(f"  Cols: {i0m}")

    """
    ============================================================
    END DEBUG: Final Divider Coordinates
    ============================================================
    """

    plt.figure()
    plt.plot(np.arange(len(Isum0)),Isum0,linewidth=2)
    ma = 1.1*Isum0.max()
    for qq in i0lasts:
        plt.plot([qq,qq], [0,ma],'r')

    plt.title('row segmentation for tier '+str(i+1))
    plt.xlabel('row'); plt.ylabel('sum along columns')
    plt.show(block=False)

    plt.figure()
    plt.plot(np.arange(len(Isum1)),Isum1,linewidth=2)
    ma = 1.1*Isum1.max()
    for qq in i1lasts:
        plt.plot([qq,qq], [0,ma],'r')

    plt.title('column segmentation for tier '+str(i+1))
    plt.xlabel('column'); plt.ylabel('sum along rows')
    plt.show(block=False)

    print(i, 'n_row_dividers', len(i1m), 'n_col_dividers', len(i0m))

    if len(i1m) < expected_row_dividers or len(i0m) < expected_col_dividers:
        raise RuntimeError("Automatic segmentation did not find enough row/column dividers for this tier.")
    
    # Estimate spacing between dividers.
    #
    # REVIEW:
    # This is only used if fullboarder is False.
    # Since fullboarder is currently hard-coded True, this spacing calculation
    # may not currently affect anything.
    
    spacing = 3+np.min([
        np.min(i1m[1:]-i1m[0:-1]),
        np.min(i0m[1:]-i0m[0:-1])
    ])

    fullboarder = True
    if fullboarder==True:
        colstart = 0
        colend = Im.shape[1]-1
        rowstart = 0
        rowend = Im.shape[0]-1
    else:
        colstart = np.max((0,i0m[0]-spacing))
        colend   = np.min((Im.shape[1]-1,i0m[-1]+spacing))
        rowstart = np.max((0,i1m[0]-spacing))
        rowend   = np.min((Im.shape[0]-1,i1m[-1]+spacing))

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
        imdisp = rotate(imdisp, ang2rot, preserve_range=True)
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

    if LOCAL_SLICES and PLOTTING :
        draw_boxes(imdisp,drawrow,drawcol,title = 'segmentation for tier '+str(i+1))

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

