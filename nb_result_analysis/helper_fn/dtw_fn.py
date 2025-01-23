import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
from dtaidistance import dtw_ndim, dtw


class MyDist(dtw.DTWSettings):
    @staticmethod
    def inner_dist(x, y):
        return np.abs(x - y) ** 2

    @staticmethod
    def result(d):
        return d

    @staticmethod
    def inner_val(x):
        return x


def dtw_warping_amount(gt_track, track, len_normalize=True, penalty=None):
    """
    Outputs the DTW amount between the the track and the ground truth track
    for both channels separately
    """

    gt_track = gt_track.astype(np.double)
    track = track.astype(np.double)

    tr_len = track.shape[0]
    path_g = dtw.warping_path_fast(track[:, 0], gt_track[:, 0], penalty=penalty)
    path_r = dtw.warping_path_fast(track[:, 1], gt_track[:, 1], penalty=penalty)
    wa_g = dtw.warping_amount(path_g)
    wa_r = dtw.warping_amount(path_r)

    if len_normalize:
        wa_g /= tr_len
        wa_r /= tr_len

    return wa_g, wa_r


def dtw_warping_dist(gt_track, track, len_normalize=False, penalty=0.1):
    """
    Outputs the DTW distance
    """

    gt_track = gt_track.astype(np.double)
    track = track.astype(np.double)

    tr_len = track.shape[0]
    dist = dtw_ndim.distance_fast(track, gt_track, penalty=penalty)

    if len_normalize:
        dist /= tr_len

    return dist


def dtw_dist_loop(gt_tracks, tracks, len_normalize=False, penalty=None):
    """
    Outputs the DTW distance between the the track and the ground truth track
    for both channels separately
    """
    n_tracks = len(tracks)
    dist_gr = np.zeros(n_tracks)

    for i in range(n_tracks):
        dist_gr[i] = dtw_warping_dist(
            gt_tracks[i], tracks[i], len_normalize=len_normalize, penalty=penalty
        )

    return dist_gr
