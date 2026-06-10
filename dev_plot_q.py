#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ------------------------------------------------------------
# Usage:
#
# python dev_plot_q.py path/to/subvolume.npz
#
# Example:
# python dev_plot_q.py test_data_bonefrags_CT_2/bonefrags_scan2_AMAAZE_outputs/bonefrags_scan2_subvolume.npz
# ------------------------------------------------------------

if len(sys.argv) != 2:
    print("Usage:")
    print("    python dev_plot_q.py <subvolume.npz>")
    sys.exit()

npzfile = sys.argv[1]

zwindow = int(input("What zwindow was used to make this subvolume?\n> "))

print(f"Loading {npzfile}")

data = np.load(npzfile)
vol = data["vol"]

# Convert reduced indices back to original slice numbers
z_original = np.arange(vol.shape[0]) * zwindow

print(f"Volume shape: {vol.shape}")

# ------------------------------------------------------------
# Reproduce q calculation from 03_segment.py
# ------------------------------------------------------------

q = -np.mean(np.mean(vol, axis=1), axis=1)

# Detect peaks using the current workflow settings
peaks, props = find_peaks(
    q,
    width=10,
    prominence=0
)

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(z_original, q, lw=2, label="q")

if len(peaks) > 0:
    plt.plot(
        z_original[peaks],
        q[peaks],
        "ro",
        label="Detected peaks"
    )

for p in peaks:
    plt.text(
        z_original[p],
        q[p],
        str(z_original[p]),
        fontsize=8,
        ha="center",
        va="bottom"
    )

plt.xlabel("Original z slice")
plt.ylabel("q")
plt.title(f"Tier detection signal (zwindow = {zwindow})")
plt.legend()

plt.tight_layout()
plt.show()

print()
print(f"Detected {len(peaks)} peaks")
print("Peak locations (reduced):", peaks)
print("Peak locations (original):", z_original[peaks])