import numpy as np


def inverse_log_min_max(y, eps=0.01, log_base=np.e):
    """
    This function takes the output of the log_min_max function and returns the original value
    Used to find biological checkpoints in the fucci signal
    """

    # General formula for any base using change of base formula
    x = ((1 + eps) / eps) ** (y * np.log(log_base) / np.log(np.e)) * eps - eps

    return x


def find_crossing_green(x, y, th=0.03, th_phase=0.15, return_idx=False):
    """
    Return the x-coordinate of the 'meaningful' threshold crossing.
    This is used to find the G1/S transitions.

    Logic:
    1. If y is above threshold for all x in [0, th_phase],
       return 0. (We say the crossing happened at 0.)
    2. Otherwise, find the first time y >= th for x > 0.15.
       If none found, return None.
    """

    # 1) Check if y is >= th *throughout* 0 <= x <= 0.15
    #    That means for all indices where x <= 0.15, y >= th.
    #    We find the indices up to 0.15, and see if y is always above threshold.
    in_early_region = np.where(x <= th_phase)[0]
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


def find_crossing_green2(x, y, th=0.1, th_phase=0.4, return_idx=False):
    """
    Return the x-coordinate of the 'meaningful' threshold crossing.
    FOR DRUGGED CELLS
    """

    crossing_indices = np.where((x > th_phase) & (y >= th))[0]
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
    tr_phase_green: float = 0.15,
    tr_red: float = 0.02,
    drug: bool = False,
    smooth_window: int = None,
):
    """
    Find the crossing points for the green and red signals.
    """
    n_tracks = len(tracks)
    crossings = np.zeros((n_tracks, 2))
    idx_crossings = np.zeros((n_tracks, 2))
    return_idx = True

    for i in range(n_tracks):
        track = tracks[i]
        if smooth_window is not None:
            track[:, 0] = np.convolve(
                track[:, 0], np.ones(smooth_window) / smooth_window, mode="same"
            )
            track[:, 1] = np.convolve(
                track[:, 1], np.ones(smooth_window) / smooth_window, mode="same"
            )

        fucci_green = inverse_log_min_max(track[:, 0])
        fucci_red = inverse_log_min_max(track[:, 1])

        if drug:
            crossing_green, idx_green = find_crossing_green2(
                taus[i],
                fucci_green,
                tr_green,
                return_idx=return_idx,
                th_phase=tr_phase_green,
            )
        else:
            crossing_green, idx_green = find_crossing_green(
                taus[i], fucci_green, tr_green, return_idx=return_idx
            )
        crossing_red, idx_red = find_crossing_red(
            taus[i], fucci_red, tr_red, return_idx=return_idx
        )
        crossings[i] = crossing_green, crossing_red
        idx_crossings[i] = idx_green, idx_red

    return crossings, idx_crossings
