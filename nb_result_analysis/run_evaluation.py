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
remove_mod_col = True

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

        # checkpoint errors in minutes
        crossing_points, idx = find_crossing_points(
            taus, tracks, drug=drug, tr_green=tr_green
        )
        delta_crossing_points = np.abs(crossing_points - crossing_points_gt)
        delta_idx = np.abs(idx - idx_gt)
        t_g, t_r = np.nanmean(delta_idx, axis=0) * 5

        # R2
        r2 = r2_track(gt_tracks, tracks).mean(axis=0)

        if not drug:
            new_row = pd.DataFrame(
                [
                    {
                        "head": head,
                        "modality": mod,
                        "L^1_{green}": l1_mean_error[0],
                        "L^1_{red}": l1_mean_error[1],
                        "\Delta t_g": t_g,
                        "\Delta t_r": t_r,
                        "R^2_{green}": r2[0],
                        "R^2_{red}": r2[1],
                    }
                ]
            )

        else:
            new_row = pd.DataFrame(
                [
                    {
                        "head": head,
                        "modality": mod,
                        "L^1_{green}": l1_mean_error[0],
                        "L^1_{red}": l1_mean_error[1],
                        "R^2_{green}": r2[0],
                        "R^2_{red}": r2[1],
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

print(f"bf{drug_str} \n")
print(generate_latex_with_bolding(df_bf, drug=drug))
print("\n")
print(f"h2b{drug_str} \n")
print(generate_latex_with_bolding(df_h2b, drug=drug))


# latex_table_bf = df_bf.style.format(precision=3).to_latex(hrules=True)
# latex_table_h2b = df_h2b.style.format(precision=3).to_latex(hrules=True)
# print("bf \n")
# print(latex_table_bf)
# print("\n")
# print("h2b \n")
# print(latex_table_h2b)


# df_bf.to_csv(f"../data/tables/df_bf{drug_str}.csv")
# df_h2b.to_csv(f"../data/tables/df_h2b{drug_str}.csv")
