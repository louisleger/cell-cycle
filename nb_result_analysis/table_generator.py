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


drug = True
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
heads = ["mlp", "cnn", "lstm", "xtransformer-causal", "mamba", "xtransformer"]


df = pd.DataFrame()
for mod in modalities:
    for head in heads:
        tracks = get_data(data_path_, mod, head)

        # L1 error
        l1_errors = mean_track_error(gt_tracks, tracks, metric="L1", av_channels=False)
        l1_mean_error = l1_errors.mean(axis=0)

        # CHECK BUGS
        crossing_points, idx = find_crossing_points(
            taus, tracks, drug=drug, tr_green=tr_green
        )
        delta_crossing_points = np.abs(crossing_points - crossing_points_gt)
        delta_idx = np.abs(idx - idx_gt)
        t_g, t_r = np.nanmean(delta_idx, axis=0) * 5

        # R2
        r2 = r2_track(gt_tracks, tracks).mean(axis=0)

        # dtw distance
        dtw_dist = dtw_dist_loop(gt_tracks, tracks, len_normalize=True).mean(axis=0)

        if not drug:
            new_row = pd.DataFrame(
                [
                    {
                        "head": head,
                        "modality": mod,
                        "L^1_{1}": l1_mean_error[0],
                        "L^1_{2}": l1_mean_error[1],
                        "\Delta t_{1}": t_g,
                        "\Delta t_{2}": t_r,
                        "R^2_{1}": r2[0],
                        "R^2_{2}": r2[1],
                        "DTW_{1}": dtw_dist[0],
                        "DTW_{2}": dtw_dist[1],
                    }
                ]
            )

        else:
            new_row = pd.DataFrame(
                [
                    {
                        "head": head,
                        "modality": mod,
                        "L^1_{1}": l1_mean_error[0],
                        "L^1_{2}": l1_mean_error[1],
                        "R^2_{1}": r2[0],
                        "R^2_{2}": r2[1],
                        "DTW_{1}": dtw_dist[0],
                        "DTW_{2}": dtw_dist[1],
                    }
                ]
            )

        df = pd.concat([df, new_row])

df
df_bf = df[df["modality"] == "bf"]
df_h2b = df[df["modality"] == "h2b"]

df_bf.index = df_bf["head"]
df_h2b.index = df_h2b["head"]

df_bf = df_bf.drop(columns=["head"])
df_h2b = df_h2b.drop(columns=["head"])

if remove_mod_col:
    df_bf = df_bf.drop(columns=["modality"])
    df_h2b = df_h2b.drop(columns=["modality"])


if drug:
    drug_str = " drug"
else:
    drug_str = ""


dfl_bf = generate_latex_with_bolding(df_bf, drug=drug, return_df=True)
dfl_h2b = generate_latex_with_bolding(df_h2b, drug=drug, return_df=True)

# erge the 2 df
dfl = pd.concat([dfl_bf, dfl_h2b], axis=0)

if not drug:
    print(f"bio table{drug_str} \n")
    bio_df = dfl[["modality", "\Delta t_{1}", "\Delta t_{2}"]]
    print(bio_df.to_latex(index=False, escape=False))

print(f"non bio table{drug_str} \n")
non_bio_df = dfl[
    ["modality", "L^1_{1}", "L^1_{2}", "R^2_{1}", "R^2_{2}", "DTW_{1}", "DTW_{2}"]
]
print(non_bio_df.to_latex(index=False, escape=False))
