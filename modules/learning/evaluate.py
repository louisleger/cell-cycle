"""

Welcome to evaluation script. In this script we define the evaluation class, which allows you to evaluate a model's performance on a dataset.
Similar to sklearn's objects like PCA(), you declare an Evaluation class and fit the model to a dataset 

Usage:
eval = Evaluation()
eval.fit(path_to_data, model, img_channels, smoothing = True)
print(eval.summary())

"""

import os
import torch
import numpy as np
import pandas as pd
from modules.utils import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    ConfusionMatrixDisplay,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
metric_functions = {
    "\ell_1": mean_absolute_error,
    "\ell_2": mean_squared_error,
    "W_d": nd_wasserstein,
    "Accuracy": tolerance_accuracy,
}


class Evaluation:
    # Initialization
    def __init__(self):
        self.prediction_df = pd.DataFrame()

    # Method to take a random slice of a track with fixed length
    def random_slice(self, x, y, slice_len):
        start = np.random.randint(0, y.shape[0] - slice_len)
        return x[:, start : start + slice_len], y[start : start + slice_len]

    def fit(
        self,
        dataset_directory,
        model,
        img_channels=[1],
        smoothing=False,
        subset=None,
        slice_len=0,
    ):
        # Define image directories and get list of cell track files
        img_directory = dataset_directory + "images/"
        label_directory = dataset_directory + "labels/"
        self.cells = os.listdir(img_directory)[:subset]
        self.slice_len = slice_len
        # Initialize pandas cells so we can store tracks inside
        self.prediction_df["CELL_ID"] = self.cells
        self.prediction_df["y"] = [None] * len(self.cells)
        self.prediction_df["y_hat"] = [None] * len(self.cells)
        model.to(DEVICE)
        for idx, cell in enumerate(tqdm(self.cells)):
            # Get input with specific image channels and groundtruth FUCCI signal
            x = (
                torch.tensor(
                    np.load(img_directory + cell, allow_pickle=True),
                    dtype=torch.float32,
                )[:, img_channels, :, :]
                .unsqueeze(0)
                .to(DEVICE)
            )
            y = np.load(label_directory + cell, allow_pickle=True).reshape(2, -1).T

            # Take a random slice of track
            if slice_len > 0 and slice_len < y.shape[0]:
                x, y = self.random_slice(x, y, slice_len)

            # Get groundtruth and prediction
            y_hat = np.array(model(x).squeeze(dim=0).cpu().tolist())

            # Temporally smooth prediction
            if smoothing:
                y_hat = multi_layer_moving_average(y_hat)

            # Save predictions and performance metrics
            for metric, function in metric_functions.items():
                self.prediction_df.loc[idx, metric] = function(y, y_hat)
            self.prediction_df.at[idx, "y"] = y
            self.prediction_df.at[idx, "y_hat"] = y_hat

    # Print a summary of the average prediction performance
    def summary(self):
        statement = f"{70*'-'}\nModel Performance:\n"
        for metric in list(metric_functions.keys()):
            statement += f"{metric}: {self.prediction_df[metric].mean():.3f} \u00B1 {self.prediction_df[metric].std():.3f}\n"
        return statement

    # Visualize these predictions, darker colors are the prediction
    def visualize_predicted_tracks(self, number, per_row=3):
        fig, ax = plt.subplots(
            number // per_row, per_row, figsize=(7 * per_row, 2.5 * number // per_row)
        )
        ax = ax.flatten()
        for idx in range(len(ax)):
            track = self.prediction_df.loc[idx, :]
            ax[idx].plot(track["y"][:, 0], "green", track["y"][:, 1], "red")
            ax[idx].plot(
                track["y_hat"][:, 0], "darkgreen", track["y_hat"][:, 1], "darkred"
            )
            ax[idx].set_title(self.prediction_df.iloc[idx, 0])

            performance = "- Performance -\n"
            for metric in list(metric_functions.keys()):
                performance += (
                    f"${metric}=${self.prediction_df.loc[idx, metric]:.3f}" + "\n"
                )
            bbox_args = dict(boxstyle="round", facecolor="whitesmoke", alpha=0.15)
            ax[idx].text(
                1.03,
                0.98,
                performance,
                transform=ax[idx].transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=bbox_args,
            )
        ax[0].legend(["$F_G$", "$F_R$", "$\widehat{F_G}$", "$\widehat{F_R}$"])
        plt.tight_layout()
        plt.show()

    # Track Normalized Time Error Profiling:
    def normalized_time_error(self, tau_steps=10):
        # Define which tau values to look at
        tau = np.linspace(0, 1, tau_steps + 1)
        errors = []
        # Iterate through predictions
        for idx, prediction in self.prediction_df.iterrows():
            # Get Prediction L1 error at specific track normalized times, averaged over output dim
            error_at_tau = np.abs(prediction["y"] - prediction["y_hat"])[
                (tau * prediction["y"].shape[0]).astype(int).tolist()[:-1] + [-1]
            ]
            errors.append(error_at_tau)
        # Format nice return dataframe
        return np.array(errors)

    def get_discrete_labels(self, y):
        # Get Arbritrary Phase Transitions
        transitions = {}
        frames = np.arange(len(y))
        transitions["G1-S"] = frames[y[:, 1] == np.max(y[:, 1])][0]
        transitions["S-G2"] = frames[y[:, 0] > 0.9 * np.max(y[:, 0])][0]
        transitions["G2-M"] = frames[y[:, 0] == np.max(y[:, 0])][0]
        # Return labels per frame encoded as = {0: 'M', 1: 'G1', 2: 'S', 3: 'G2'}
        return (
            pd.Series(frames)
            .apply(
                lambda frame: (
                    1
                    if frame < transitions["G1-S"]
                    else (
                        2
                        if frame < transitions["S-G2"]
                        else 3 if frame < transitions["G2-M"] else 0
                    )
                )
            )
            .values
        )

    def discrete_matrix(self):
        # Get Confusion Matrix for discrete labels and traditional Accuracy
        self.prediction_df["y_discrete"] = self.prediction_df["y"].apply(
            lambda y: self.get_discrete_labels(y)
        )
        self.prediction_df["y_hat_discrete"] = self.prediction_df["y_hat"].apply(
            lambda y: self.get_discrete_labels(y)
        )
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(
            y_true=np.concatenate(self.prediction_df["y_discrete"].values),
            y_pred=np.concatenate(self.prediction_df["y_hat_discrete"].values),
            normalize="true",
            display_labels=["M", "G1", "S", "G2"],
            ax=ax,
            cmap=plt.cm.Blues,
        )
        ax.set_title("Confusion Matrix of Discrete Labels")
        plt.tight_layout()
        plt.show()

    def onset_delays(self):
        for signal in ["", "_hat"]:
            self.prediction_df[f"G1-S{signal}"] = self.prediction_df[
                f"y{signal}"
            ].apply(lambda y: np.arange(len(y))[y[:, 1] == np.max(y[:, 1])][0])
        return self.prediction_df.apply(
            lambda p: np.abs(p["G1-S"] - p["G1-S_hat"]), axis=1
        ).mean()


import torch
import torch.nn as nn
from tqdm import tqdm
from modules.learning.train import track_dataset
import pandas as pd
import numpy as np
import umap
from sklearn.decomposition import PCA
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from einops.layers.torch import Reduce

DATA_PATH = "/media/maxine/c8f4bcb2-c1fe-4676-877d-8e476418f5e5/0-RPE-cell-timelapse/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def plot_umap(
    u,
    data=None,
):
    if data is None:
        data = np.zeros(len(u))
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(u[:, 0], u[:, 1], s=0.5, c=data)
    plt.show()


def get_latent_space(
    directory, model, img_channels, n_pca_components=1, seed=5, track_mean=False
):
    # Set up dataset, dataloader and remove prediction head to get internal representations
    model.prediction_head = nn.Identity()
    dataset = track_dataset(directory, img_channels)
    loader = DataLoader(
        dataset, batch_size=1, pin_memory=True, num_workers=1, shuffle=True
    )
    if track_mean:
        model.prediction_head = Reduce("b t e -> b 1 e", "mean")
    model.eval()
    model.to(DEVICE)
    # Go through dataset and save internal representations in representations df
    representations_df = pd.DataFrame()
    representations_df["z"] = [None] * len(dataset)
    representations_df["tau"] = [None] * len(dataset)
    start_time = time.time()
    with torch.no_grad():
        for idx, (name, imgs, la) in enumerate(tqdm(loader)):
            z = model(imgs.to(DEVICE))
            representations_df.at[idx, "z"] = z.cpu().numpy()[0]
            representations_df.at[idx, "tau"] = np.linspace(0, 1, imgs.shape[1])
            representations_df.at[idx, "name"] = name[0]
    print("Got internal representations in ", round(time.time() - start_time, 2), "s")
    start_time = time.time()

    # Run umap and plot it
    z_points = np.concatenate(representations_df["z"].values, axis=0)  # [:int(3e5)]
    # Run PCA before, common practice
    pca = PCA(n_components=n_pca_components, random_state=seed)
    explained_var = None
    if n_pca_components > 2:
        z_points = pca.fit_transform(z_points)
        print(z_points.shape)
        explained_var = pca.explained_variance_ratio_
    # UMAP execution
    mapper = umap.UMAP(n_neighbors=10, min_dist=0.1, random_state=seed, n_jobs=1)
    u1 = mapper.fit_transform(z_points)
    print("Got Manifold Projection in", round(time.time() - start_time, 2), "s")
    return representations_df, u1, explained_var
