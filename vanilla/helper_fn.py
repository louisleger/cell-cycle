import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
from dtaidistance import dtw_ndim, dtw


def norm_time(track):
    return np.arange(len(track)) / len(track)


def plot_fucci(track, time="standard", delta_t=5, label=None):
    if time == "standard":
        t = np.arange(0, track.shape[0]) * delta_t
        plt.xlabel("Time [min]")
    elif time == "normalized":
        t = np.arange(0, track.shape[0]) / track.shape[0]
        plt.xlabel("normalized time")
    plt.plot(t, track[:, 0], label="fucci green", c="g")
    plt.plot(t, track[:, 1], label="fucci red", c="r")
    plt.legend()


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


# define the function that will reconstruct the signal, using the coefficients
def reconstruct_signal(x, coeffs, n_harmonics):
    m = len(x)
    A = np.ones((m, 1 + 2 * n_harmonics))  # include column for intercept

    for k in range(1, n_harmonics + 1):
        A[:, 2 * k - 1] = np.cos(2 * np.pi * k * x)  # cosine terms
        A[:, 2 * k] = np.sin(2 * np.pi * k * x)  # sine terms

    y_fit = A.dot(coeffs)
    return y_fit


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
