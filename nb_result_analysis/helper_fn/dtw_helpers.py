import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
from dtaidistance import dtw_ndim, dtw


def norm_time(track):
    return np.arange(len(track)) / len(track)


def linear_regression(x, y, n_harmonics=10):

    def design_matrix(tau, n_harmonics=100):
        """
        Function that returns the design matrix for a the fourier basis
        """
        design = np.zeros((len(tau), 2 * n_harmonics + 1))
        k_values = np.linspace(1, n_harmonics, n_harmonics)
        # compute the fourier basis
        cosines = np.cos(2 * np.pi * k_values * tau[:, None])
        sines = np.sin(2 * np.pi * k_values * tau[:, None])

        design[:, 0] = 1
        design[:, 1::2] = cosines
        design[:, 2::2] = sines

        return design

    X = design_matrix(x, n_harmonics=n_harmonics)

    # Solve the linear least squares problem
    coeffs, _, _, _ = lstsq(X, y)
    # Reconstruct the signal from the coefficients
    y_fit = X.dot(coeffs)
    return coeffs, y_fit


def normalize_ref(path, ref_len):
    """
    Normalizes the reference column (first column) of a path
    """
    path = path.astype(float)
    path[:, 0] = path[:, 0] / ref_len
    return path


def phase_map(path, ref_len):
    """
    This function takes the output of the warping path of ref (or phase) and input:
    - normalizes the reference path (which is the first column of the path)
    and supposed to span the entire cycle (CC phase)
    - calculates the phase map of the second column of the path, making sure
    that every time the input is mapped to more than one phase value, the mean
    of the phase values is returned

    Returns:
    - the phase of each input value
    """

    path = normalize_ref(path, ref_len)
    phase, input = path[:, 0], path[:, 1]
    input_u = np.unique(input)

    out = np.zeros(input_u.shape)
    for i, val in enumerate(input_u):
        out[i] = phase[input == val].mean()
    return out


###########
# DWT functions
###########


def align_to_reference(track, ref, window_size=None, penalty=None, psi=None):
    """
    Aligns a track to a reference signal using dynamic time warping
    """
    if window_size is None:
        window_size = len(ref)
    d, wpath = dtw_ndim.warping_paths_fast(
        track, ref, window=window_size, psi=psi, penalty=penalty
    )
    path = np.array(dtw.best_path(wpath)).astype(float)
    return path


def track2phase(track, ref, window_size=None, penalty=None, psi=None):
    """
    Aligns a track to a reference signal and normalizes the reference signal, which is the phase
    """
    path = align_to_reference(
        track, ref, window_size=window_size, penalty=penalty, psi=psi
    )
    # normalize the ref column
    path[:, 1] = path[:, 1] / len(ref)
    return path


# I want a function that does this for every entry of  a vector
def phase2class_(phase, borders=[0.45, 0.68, 0.93]):
    """
    Classifies a phase into one of 4 classes:
    class 0 = G1
    class 1 = S
    class 2 = G2
    class 3 = M
    Inputs;
    Phase: array of size (n, 2) where the first column is the time (index of timepoint in track)
    and the second column is the phase (number between 0 and 1)
    borders: array of size 3, where the first element is the border between G1 and S, the second between S and G2
    and the third between G2 and M
    """
    if phase < borders[0]:
        return 0
    elif phase < borders[1]:
        return 1
    elif phase < borders[2]:
        return 2
    else:
        return 3


def phase2class(phase, borders=[0.45, 0.68, 0.93]):
    """
    Classifies a phase into one of 4 classes:
    class 0 = G1
    class 1 = S
    class 2 = G2
    class 3 = M
    Inputs;
    Phase: array of size (n, 2) where the first column is the time (index of timepoint in track)
    and the second column is the phase (number between 0 and 1) output of the function track2phase
    borders: array of size 3, where the first element is the border between G1 and S, the second between S and G2
    and the third between G2 and M
    """
    class_vec = phase.copy()
    class_vec[:, 1] = np.array([phase2class_(p, borders) for p in phase[:, 1]])
    return class_vec


def class_duration(class_vec):
    """
    Given a vector of classes, returns the duration of each class
    Inputs:
    class_vec: array of size (n, 2) where the first column is the time (index of timepoint in track)
    It is the output of the function phase2class
    """
    _, durations = np.unique(class_vec[:, 1], return_counts=True)
    return durations


def vanilla_fn(
    tau,
    coeffs,
):

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


# def dtw_dist_gr(gt_track, track, penalty=None):
#     """
#     Outputs the DTW distance between the the track and the ground truth track
#     for both channels separately
#     """

#     gt_track = gt_track.astype(np.double)
#     track = track.astype(np.double)
#     dist_g = dtw.distance_fast(track[:, 0], gt_track[:, 0], penalty=penalty)
#     dist_r = dtw.distance_fast(track[:, 1], gt_track[:, 1], penalty=penalty)

#     return dist_g, dist_r
