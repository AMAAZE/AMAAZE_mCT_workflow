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

from utils import *

slicepath = os.path.normpath(slicepath)

if not os.path.isdir(slicepath):
    raise RuntimeError("The provided slicepath was not found.")

if not os.path.exists(layoutfile): 
    raise RuntimeError("The layoutfile was not found.")

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
    
AUTO_ROT, AUTO_SEG, INPUT_ROWS, INPUT_COLS, TIER_THRESH, INVERT, THRESH0, THRESH1 = read_hyperparameters(
    scan_num, directory=scanpath
)

REDO_TIERS=True
REDO_ISO = True
PLOTTING = True

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

rowsz = rowrng[1]-rowrng[0]
colsz = colrng[1]-colrng[0]

if INVERT:
    vol = vol.max() - vol

x = pd.read_csv(layoutfile).fillna(0)
x = x.to_numpy()
x = x[x[:,0]==scan_num,1:].copy()
    
if len(x) == 0:
    raise ValueError(
        f'Scan {scan_num} not found in layout file. '
        'Check that the CSV includes this scan and follows the required format.'
    )

n_tiers = int(x[:,0].max())
dim1 = int(x[:,1].max())
dim2 = x.shape[1]-2

scan_layout = np.zeros((n_tiers,dim1,dim2),object)
for i in range(n_tiers):
    scan_layout[i,:,:] = x[x[:,0]==i+1,2:]

mask = scan_layout!=0 #cells to extract from
tier_mask = np.sum(np.sum(mask,1),1)>0

# Step 1: segment vertically
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

ranges = []
for i in range(len(ex)-1):
    ranges.append([ex[i],ex[i+1]])

ranges = ranges[-1::-1]
tiers = np.arange(n_tiers)[tier_mask]
ranges = [ranges[i] for i in tiers]

SLICES = []
for i in range(len(ranges)):
    SLICES.append(vol[ranges[i][0]:ranges[i][1],:,:])

I = [x.mean(0) for x in SLICES]

fig = plt.figure(figsize=(10,4))
plt.hist(vol.flatten(), bins=500)
plt.title("voxel value histogram")
plt.ylabel("frequency")
plt.xlabel("voxel value")

iso_override = iso_thresholds_override

if iso_override is not None:
    t1t2t3 = np.array(iso_override).astype(int)
    print(f"Using ISO threshold override from JSON: {t1t2t3}")

elif REDO_ISO:
    t1t2t3 = get_iso_thresholds_from_voxel_probe(
        slicepath=slicepath,
        slice_index_fraction=slice_index_fraction,
        transpose_preview=transpose_preview,
        ang2rot=ang2rot,
        rowrng=rowrng,
        colrng=colrng,
        n_clicks=voxel_probe_n_clicks
    )

    update_param(scan_num, "ISO_THRESHOLDS", t1t2t3.tolist(), directory=scanpath)

else:
    t1t2t3 = get_parameter(scan_num, "ISO_THRESHOLDS", directory=scanpath)

EXTRACTS = []

for i in range(len(tiers)):
    tier = tiers[i]; # Im = I[i].T
    si = SLICES[i]

    if AUTO_ROT:
        ''' same params go into id_cardboard:'''
        angi,Im = autorot2(
            si,
            t1=t1t2t3[0],
            t2=t1t2t3[1],
            t3=t1t2t3[2],
            title="rotation for tier "+str(i+1)
        )

    else:
        ''' same params go into autorot2:'''
        Im = id_cardboard(
            si,
            t1=t1t2t3[0],
            t2=t1t2t3[1],
            t3=t1t2t3[2]
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

    Isum0 = np.sum(Im,0)
    Isum1 = np.sum(Im,1)

    if (row_dividers_override is not None) or (col_dividers_override is not None):
        x0 = np.zeros_like(Isum0, dtype=int)
        x1 = np.zeros_like(Isum1, dtype=int)
    elif AUTO_SEG == True:
        x0 = auto_seg(Isum0, n_dividers_col)
        x1 = auto_seg(Isum1, n_dividers_row)
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

    if INPUT_ROWS:
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

    if len(i1m) < 2 or len(i0m) < 2:
        raise RuntimeError("Automatic segmentation did not find enough row/column dividers for this tier.")
    
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

    rowcolrng = np.vstack((row[rowi],row[rowi+1],col[coli],col[coli+1])).T
    namesi = layouti[maski,None]

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

    drawrow = ((rowsz / vol.shape[1]) * (rowcolrng[:,0:2])).astype(int)
    drawcol = ((colsz / vol.shape[2]) * (rowcolrng[:,2:4])).astype(int)

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

pd.DataFrame(E).to_csv(
    os.path.join(scanpath, f"CT{scan_num}.csv"),
    header=False,
    index=False
)

