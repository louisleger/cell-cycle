import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys
from sklearn.metrics import r2_score

from helper_fn import *

sys.path.append("..")
# from modules.utils import paper_style

# loading GT stuff

current_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = "../data"
path = os.path.join(current_dir, relative_path)
data_path = os.path.join(path, "test/")
data_drug_path = os.path.join(path, "test_drug/")
gt_path = os.path.join(path, "GT/")
gt_drug_path = os.path.join(path, "GT_drug/labels/")


############
# drug switch
############


drug = False
remove_mod_col = False

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

if drug == False:
    tr_green = 0.035
    tr_phase_green = 0.15
elif drug == True:
    tr_green = 0.14
    tr_phase_green = 0.4

crossing_points_gt, idx_gt = find_crossing_points(
    taus, gt_tracks, drug=drug, tr_green=tr_green
)

track_lengths = [track.shape[0] for track in gt_tracks]

modalities = ["bf", "h2b"]
df_metrics = pd.DataFrame()  # For all metrics except "\Delta t_{}"
df_delta_t = pd.DataFrame()  # For "\Delta t_{}" (only when drug=False)

for mod in modalities:
    for i, head in enumerate(heads):
        tracks = get_data(data_path_, mod, head)

        # L1 error
        l1_errors = mean_track_error(gt_tracks, tracks, metric="L1", av_channels=False)
        l1_mean_error = l1_errors.mean(axis=0)
        l1_std_error = l1_errors.std(axis=0)

        # CHECK BUGS
        crossing_points, idx = find_crossing_points(
            taus, tracks, drug=drug, tr_green=tr_green
        )
        delta_crossing_points = np.abs(crossing_points - crossing_points_gt)
        delta_idx = np.abs(idx - idx_gt)
        t_g, t_r = np.nanmean(delta_idx, axis=0) * 5
        sigma_t_g, sigma_t_r = np.nanstd(delta_idx, axis=0) * 5

        # R2
        r2 = r2_track(gt_tracks, tracks)
        r2_mean = r2.mean(axis=0)
        r2_std = r2.std(axis=0)

        # dtw distance
        dtw_dist = dtw_dist_loop(gt_tracks, tracks, len_normalize=False, penalty=0.1)
        dtw_dist_mean = dtw_dist.mean(axis=0)
        dtw_dist_std = dtw_dist.std(axis=0)

        # Common metrics DataFrame
        new_metrics_row = pd.DataFrame(
            [
                {
                    "head": head_names[i],
                    "modality": mod,
                    "L^1_{1}": l1_mean_error[1],
                    "L^1_{2}": l1_mean_error[0],
                    "R^2_{1}": r2_mean[1],
                    "R^2_{2}": r2_mean[0],
                    "DTW": dtw_dist_mean,
                }
            ]
        )
        df_metrics = pd.concat([df_metrics, new_metrics_row])

        # WT-specific metrics (only when drug=False)
        if not drug:
            new_delta_t_row = pd.DataFrame(
                [
                    {
                        "head": head_names[i],
                        "modality": mod,
                        "\Delta t_{1}": t_g,
                        "\Delta t_{2}": t_r,
                    }
                ]
            )
            df_delta_t = pd.concat([df_delta_t, new_delta_t_row])

# Set index and drop columns as needed
df_metrics.index = df_metrics["head"]
df_metrics = df_metrics.drop(columns=["head"])

if not drug:
    df_delta_t.index = df_delta_t["head"]
    df_delta_t = df_delta_t.drop(columns=["head"])
    df_delta_t.columns = ["modality", "\Delta t_{G1/S} [min]", "\Delta t_{S/G2} [min]"]

# # Remove modality column if specified
# if remove_mod_col:
#     df_metrics = df_metrics.drop(columns=["modality"])
#     if not drug:
#         df_delta_t = df_delta_t.drop(columns=["modality"])

df_metrics = df_metrics.round(3)
if not drug:
    df_delta_t = df_delta_t.round(1)
df_metrics

# print(df_metrics.to_string())


print(generate_latex_with_bolding_3f(df_metrics, return_df=False))
if not drug:
    print(generate_latex_with_bolding_1f(df_delta_t, return_df=False))
