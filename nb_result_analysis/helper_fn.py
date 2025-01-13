import numpy as np
import os


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


def find_threshold_crossing(x, y, th):
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
            return 0

    # 2) Otherwise, we look for the first index i where x[i] > 0.15 and y[i] >= th
    crossing_indices = np.where((x > 0.15) & (y >= th))[0]
    if len(crossing_indices) == 0:
        return None

    # Return the x-coordinate of the first crossing
    return x[crossing_indices[0]]
