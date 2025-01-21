import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from find_checkpoints import *

# name of all heads
heads = ["mlp", "cnn", "lstm", "xtransformer-causal", "mamba", "xtransformer"]

# colors assigned to each head, can be improved
available_colors = ["#1f77b4", "brown", "purple", "#8c564b", "#9467bd", "orange"]
head_colors = {
    head: available_colors[i % len(available_colors)] for i, head in enumerate(heads)
}


def get_data(path, modality, head):
    """
    Gets data from the path, modality and head, it outputs a list of tracks

    Parameters:
    path: string, path to the data
    modality: string, modality of the data
    head: string, head of the model

    Returns:
    tracks: list of tracks
    """
    mod_path = os.path.join(path, modality)

    folders = [
        f for f in os.listdir(mod_path) if os.path.isdir(os.path.join(mod_path, f))
    ]

    # avoids the problem that xtransformer is contained in  xtransformer-causal string
    if head == "xtransformer":
        head = "xtransformer_"

    for f in folders:
        if head in f:
            head_folder = f
            break

    # pick the head folder, and get the track filenames
    head_path = os.path.join(mod_path, head_folder)
    track_filenames = os.listdir(head_path)

    # opening tracks in the first folder
    tracks = []
    for i, fn in enumerate(track_filenames):
        track = np.load(os.path.join(head_path, fn)).squeeze().T
        tracks.append(track)

    return tracks


def mean_track_error(gt, tracks, metric="L1", av_channels=False):
    """
    Takes mean error between gt and tracks,

    Parameters:
    - gt: list of ground truth tracks
    - tracks: list of tracks to compare with gt
    - metric: L1 or L2
    - av_channels: if True, takes mean over the 2 channels as well
    """

    if av_channels:
        axis = (0, 1)
    else:
        axis = 0

    if metric == "L1":
        error = [np.mean(np.abs(gt[i] - tracks[i]), axis=axis) for i in range(len(gt))]
    elif metric == "L2":
        error = [np.mean((gt[i] - tracks[i]) ** 2, axis=axis) for i in range(len(gt))]

    return np.array(error)


def r2_track(gt, tracks):
    """
    Returns the R2 score between the ground truth for all tracks and the predicted tracks
    """
    n_tracks = len(tracks)
    r2 = np.zeros((n_tracks, 2))

    for i in range(len(gt)):
        r2_green = r2_score(gt[i][:, 0], tracks[i][:, 0])
        r2_red = r2_score(gt[i][:, 1], tracks[i][:, 1])

        r2[i] = r2_green, r2_red

    return r2


def track_errors_flattened(gt_tracks, tracks):
    """
    This returns all erros in a flattened form
    without averaging over the tracks,
    called by bin_avarage_errors
    """

    n_tracks = len(tracks)
    errors_g = []
    errors_r = []
    taus = []
    for i in range(n_tracks):
        error = np.abs(tracks[i] - gt_tracks[i])
        errors_g.append(error[:, 0])
        errors_r.append(error[:, 1])
        taus.append(np.linspace(0, 1, error.shape[0]))
    errors_g_unrolled = np.concatenate(errors_g)
    errors_r_unrolled = np.concatenate(errors_r)
    taus_unrolled = np.concatenate(taus)

    return errors_g_unrolled, errors_r_unrolled, taus_unrolled


def bin_avarage_errors(errors, taus, n_bins=30):
    """
    Take a list of errors and taus, and bin them according to the taus.
    It takes the output of track_errors_flattened as input

    Parameters:
    errors: list of errors
    taus: list of taus
    n_bins: number of bins
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(taus, bins) - 1  # Bin indices (0-indexed)

    # Compute bin averages
    averaged_profile = np.array(
        [
            errors[bin_indices == i].mean() if np.any(bin_indices == i) else 0
            for i in range(n_bins)
        ]
    )
    return averaged_profile, bins


###############################################
# plotting and latex function
###############################################


def vanilla_fn(
    tau,
    coeffs=None,
):
    """
    returns vanilla_fn(tau) given the fourier coefficients
    """
    if coeffs is None:
        coeffs = np.load("../vanilla/coef_fourier.npy")

    n_harmonics = (coeffs.shape[0] - 1) // 2
    k_values = np.arange(1, n_harmonics + 1)

    # Fourier Design Matrix
    A = np.ones((tau.shape[0], 1 + 2 * n_harmonics))

    cosine_terms = np.cos(2 * np.pi * k_values[None, :] * tau[:, None])
    sine_terms = np.sin(2 * np.pi * k_values[None, :] * tau[:, None])
    A[:, 1::2] = cosine_terms
    A[:, 2::2] = sine_terms

    # Get Vanilla Prediction
    # vanilla_prediction = A @ coeffs
    vanilla_prediction = np.einsum("sp,pf->sf", A, coeffs)

    return vanilla_prediction


def plot_fucci(track, time="standard", delta_t=5, label=None):
    if time == "standard":
        t = np.arange(0, track.shape[0]) * delta_t
        plt.xlabel("Time [min]")
    elif time == "normalized":
        t = np.arange(0, track.shape[0]) / track.shape[0]
        plt.xlabel("normalized time")
    plt.plot(t, track[:, 0], label=label, c="g")
    plt.plot(t, track[:, 1], label=label, c="r")
    plt.legend()


#############################


# Function to process the DataFrame and generate LaTeX output with 3 decimal places
def generate_latex_with_bolding(df, drug):
    latex_df = df.copy()

    if drug:
        min_columns = ["L^1_{green}", "L^1_{red}"]
    else:
        min_columns = ["L^1_{green}", "L^1_{red}", "\Delta t_g", "\Delta t_r"]

    for col in min_columns:
        min_value = df[col].min()
        latex_df[col] = df[col].apply(
            lambda x: f"\\textbf{{{x:.3f}}}" if x == min_value else f"{x:.3f}"
        )

    for col in ["R^2_{green}", "R^2_{red}"]:
        max_value = df[col].max()
        latex_df[col] = df[col].apply(
            lambda x: f"\\textbf{{{x:.3f}}}" if x == max_value else f"{x:.3f}"
        )

    return latex_df.to_latex(index=True, escape=False)
