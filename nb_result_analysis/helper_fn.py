import numpy as np
import os
import matplotlib.pyplot as plt


def get_data(path, modality, head):
    """
    Gets data from the path, modality and head, it outputs a list of tracks
    """
    mod_path = os.path.join(path, modality)

    folders = [
        f for f in os.listdir(mod_path) if os.path.isdir(os.path.join(mod_path, f))
    ]

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


def mean_track_error(gt, tracks, metric="L1", av_channels=True):
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
        error = [
            np.mean(np.abs(gt[i] - track), axis=axis) for i, track in enumerate(tracks)
        ]
    elif metric == "L2":
        error = [
            np.mean((gt[i] - track) ** 2, axis=axis) for i, track in enumerate(tracks)
        ]

    return np.array(error)


def inverse_log_min_max(y, eps=0.01, log_base=np.e):
    """
    This function takes the output of the log_min_max function and returns the original value
    """

    # General formula for any base using change of base formula
    x = ((1 + eps) / eps) ** (y * np.log(log_base) / np.log(np.e)) * eps - eps

    return x


def find_crossing_green(x, y, th=0.03, return_idx=False):
    """
    Return the x-coordinate of the 'meaningful' threshold crossing.
    This is used to find the G1/S transitions.

    Logic:
    1. If y is above threshold for all x in [0, 0.15],
       return 0. (We say the crossing happened at 0.)
    2. Otherwise, find the first time y >= th for x > 0.15.
       If none found, return None.
    """

    # 1) Check if y is >= th *throughout* 0 <= x <= 0.15
    #    That means for all indices where x <= 0.15, y >= th.
    #    We find the indices up to 0.15, and see if y is always above threshold.
    in_early_region = np.where(x <= 0.15)[0]
    if len(in_early_region) > 0:
        # All y values in [0, 0.15] region
        y_early = y[in_early_region]
        if np.all(y_early >= th):
            # Then the signal never dipped below threshold in [0, 0.15]
            return 0, 0 if return_idx else 0

    # 2) Otherwise, we look for the first index i where x[i] > 0.15 and y[i] >= th
    crossing_indices = np.where((x > 0.15) & (y >= th))[0]
    if len(crossing_indices) == 0:
        return None, None if return_idx else None

    # Return the x-coordinate of the first crossing
    if return_idx:
        return x[crossing_indices[0]], crossing_indices[0]
    else:
        return x[crossing_indices[0]]


def find_crossing_red(x, y, threshold=0.02, return_idx=False):
    """
    Return the x-coordinate where the signal first drops below `threshold`
    after x > 0.5.
    If no such drop is found, return None.
    """
    # Identify all indices where x > 0.5 and y < threshold
    vanish_indices = np.where((x > 0.5) & (y < threshold))[0]
    if len(vanish_indices) == 0:
        return None, None if return_idx else None

    if return_idx:
        return x[vanish_indices[0]], vanish_indices[0]
    else:
        return x[vanish_indices[0]]


def find_crossing_points(
    taus: list,
    tracks: list,
    tr_green: float = 0.03,
    tr_red: float = 0.02,
):
    """
    Find the crossing points for the green and red signals.
    """
    n_tracks = len(tracks)
    crossings = np.zeros((n_tracks, 2))
    idx_crossings = np.zeros((n_tracks, 2))
    return_idx = True

    for i in range(n_tracks):
        fucci_green = inverse_log_min_max(tracks[i][:, 0])
        fucci_red = inverse_log_min_max(tracks[i][:, 1])

        crossing_green, idx_green = find_crossing_green(
            taus[i], fucci_green, tr_green, return_idx=return_idx
        )
        crossing_red, idx_red = find_crossing_red(
            taus[i], fucci_red, tr_red, return_idx=return_idx
        )
        crossings[i] = crossing_green, crossing_red
        idx_crossings[i] = idx_green, idx_red

    return crossings, idx_crossings


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
