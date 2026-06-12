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

mean_intensity_profile_z = -np.mean(np.mean(vol, axis=1), axis=1)

shifted_mean_intensity_profile_z = (
    mean_intensity_profile_z
    - mean_intensity_profile_z.min()
)

# Investigate threshold without applying it
thresh = 3e7 / (vol.shape[1] * vol.shape[2])

below_thresh = shifted_mean_intensity_profile_z < thresh

print(f"Threshold: {thresh}")
print(f"Points below threshold: {np.sum(below_thresh)}")

# Detect peaks using the current workflow settings
peaks, props = find_peaks(
    shifted_mean_intensity_profile_z,
    width=10,
    prominence=0
)

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(z_original, mean_intensity_profile_z, lw=2, label="mean_intensity_profile_z")

if len(peaks) > 0:
    plt.plot(
        z_original[peaks],
        mean_intensity_profile_z[peaks],
        "ro",
        label="Detected peaks"
    )

for p in peaks:
    plt.text(
        z_original[p],
        mean_intensity_profile_z[p],
        str(z_original[p]),
        fontsize=8,
        ha="center",
        va="bottom"
    )

plt.xlabel("Original z slice")
plt.ylabel("mean_intensity_profile_z")
plt.title(f"Tier detection signal (zwindow = {zwindow})")
plt.legend()

plt.tight_layout()
plt.show()

print()
print(f"Detected {len(peaks)} peaks")
print("Peak locations (reduced):", peaks)
print("Peak locations (original):", z_original[peaks])