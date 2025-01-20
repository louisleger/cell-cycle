import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys
from sklearn.metrics import r2_score

from helper_fn import *

sys.path.append("..")
from modules.utils import paper_style

paper_style()

# loading GT stuff

current_dir = ""
relative_path = "../data"
path = os.path.join(current_dir, relative_path)
# data_path = os.path.join(path, "results/")
data_path = os.path.join(path, "test/")

data_drug_path = os.path.join(path, "test_drug/")
gt_path = os.path.join(path, "GT/")
gt_drug_path = os.path.join(path, "GT_drug/labels/")

drug = False

if drug:
    data_path_ = data_drug_path
    gt_path_ = gt_drug_path
else:
    data_path_ = data_path
    gt_path_ = gt_path

gt_tracks = []
taus = []
for i, fn in enumerate(os.listdir(gt_path_)):
    track = np.load(os.path.join(gt_path_, fn)).squeeze().T
    tau = np.linspace(0, 1, track.shape[0])
    gt_tracks.append(track)
    taus.append(tau)

track_lengths = [track.shape[0] for track in gt_tracks]

modality = "bf"

head = "xtransformer"

errors_g = {}
errors_r = {}


for head in heads:

    tracks = get_data(path=data_path_, modality=modality, head=head)
    n_tracks = len(tracks)
    fucci_labels = ["green", "red"]

    errors_g_unrolled, errors_r_unrolled, taus_unrolled = track_errors_flattened(
        gt_tracks, tracks
    )

    errors_g_averaged, bins = bin_avarage_errors(errors_g_unrolled, taus_unrolled)
    errors_r_averaged, bins = bin_avarage_errors(errors_r_unrolled, taus_unrolled)

    errors_g[head] = errors_g_averaged
    errors_r[head] = errors_r_averaged


# plt.title(f"heads errors")
# plt.xlabel("phase")
# plt.ylabel("average error")
# for head in good_heads:
#     errors_g_averaged = errors_g[head]
#     errors_r_averaged = errors_r[head]

#     plt.plot(bins[:-1], errors_g_averaged, label=f"{head}", color=head_colors[head])

# plt.legend()


import matplotlib.pyplot as plt

# Create the figure and subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

# Left subplot for errors_g_averaged
ax1.set_title("Green Channel")
ax1.set_xlabel("Phase")
ax1.set_ylabel("Average Error")
for head in good_heads:
    errors_g_averaged = errors_g[head]
    ax1.plot(bins[:-1], errors_g_averaged, label=f"{head}", color=head_colors[head])

ax1.legend()
ax1.grid()

# Right subplot for errors_r_averaged
ax2.set_title("Red channel")
ax2.set_xlabel("Phase")
for head in good_heads:
    errors_r_averaged = errors_r[head]
    ax2.plot(bins[:-1], errors_r_averaged, label=f"{head}", color=head_colors[head])

ax2.legend()
ax2.grid()

# Ensure tight layout and consistent y-axis scaling
plt.tight_layout()


# save as pdf in ../plots/av_error
plt.savefig(f"../plots/av_error/av_error.pdf")
plt.show()


# plt.title(f"{modality} {head} fucci L1 errors averaged")

# plt.plot(bins[:-1], errors_g_averaged, color="green", label="green")
# plt.plot(bins[:-1], errors_r_averaged, color="red", label="red")

# plt.ylim(0, 0.5)

# plt.xlabel("phase")
# plt.ylabel("average error")

# save as pdf in ../plots/av_error
# plt.savefig(f"../plots/av_error/{modality}_{head}_fucci_errors.pdf")
