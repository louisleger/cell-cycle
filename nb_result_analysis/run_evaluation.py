import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys
from helper_fn import *

sys.path.append("..")
# from modules.utils import paper_style

# loading GT stuff

current_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = "../data"
path = os.path.join(current_dir, relative_path)
data_path = os.path.join(path, "results/")
data_drug_path = os.path.join(path, "test_drug/")
gt_path = os.path.join(path, "GT/")
gt_drug_path = os.path.join(path, "GT_drug/labels/")

drug = True

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

crossing_points_gt, idx_gt = find_crossing_points(taus, gt_tracks)

track_lengths = [track.shape[0] for track in gt_tracks]

modalities = ["bf", "h2b"]
heads = ["mlp", "cnn", "lstm", "xtransformer-causal", "mamba", "xtransformer"]

df = pd.DataFrame(
    columns=["head", "modality", "l1_error_green", "l1_error_red", "t_g", "t_r"]
)


for mod in modalities:
    for head in heads:
        tracks = get_data(data_path_, mod, head)

        l1_errors = mean_track_error(gt_tracks, tracks, metric="L1", av_channels=False)
        l1_mean_error = l1_errors.mean(axis=0)

        crossing_points, idx = find_crossing_points(taus, tracks)
        delta_crossing_points = np.abs(crossing_points - crossing_points_gt)
        delta_idx = np.abs(idx - idx_gt)
        t_g, t_r = np.nanmean(delta_idx, axis=0) * 5

        new_row = pd.DataFrame(
            [
                {
                    "head": head,
                    "modality": mod,
                    "l1_error_green": l1_mean_error[0],
                    "l1_error_red": l1_mean_error[1],
                    "t_g": t_g,
                    "t_r": t_r,
                }
            ]
        )
        df = pd.concat([df, new_row])

        print(
            f"Modality: {mod}, Head: {head}, \nMean Error: {l1_mean_error.mean():.3f} G1/S green: {t_g:.1f}min red decay {t_r:.1f}min \n "
        )

mod, head = "bf", "xtransformer"
tracks = get_data(data_path_, mod, head)

# random index
idx = 23
plt.title(f"Modality: {mod}, Head: {head}")
# plot prediction
plot_fucci(tracks[idx])
# plot gt
plot_fucci(gt_tracks[idx], label="prediction")
