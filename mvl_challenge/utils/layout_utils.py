
import numpy as np
from mlc.models.HorizonNet.misc import panostretch
import logging
from copy import deepcopy
import os
from pathlib import Path



def label_cor2ly_phi_coord(label_cor_path, shape=(512, 1024)):
    """
    Implementation taken from HorizonNet (CVPR 2019)
    https://sunset1995.github.io/HorizonNet/ 
    """
    with open(label_cor_path) as f:
        cor = np.array(
            [line.strip().split() for line in f if line.strip()], np.float32
        )
    return corners_uv2ly_phi_coord(corners=cor, shape=shape)


def corners_uv2ly_phi_coord(corners, shape=(512, 1024)):
    """
    Return an npy array of 2xW points corresponded to the
    layout boundaries defined in spherical coords (phi only).
    Similar to the output estimation of HorizonNet and HoHoNet
    Implementation taken from HorizonNet (CVPR 2019)
    https://sunset1995.github.io/HorizonNet/
    """
    # Corner with minimum x should at the beginning
    corners = np.roll(corners[:, :2], -2 * np.argmin(corners[::2, 0]), 0)

    H, W = shape
    # Detect occlusion
    assert (np.abs(corners[0::2, 0] - corners[1::2, 0]) > W / 100).sum() == 0
    assert (corners[0::2, 1] > corners[1::2, 1]).sum() == 0

    bon = cor_2_1d(corners, H, W)

    return bon


def sort_xy_filter_unique(xs, ys, y_small_first=True):
    """
    Implementation taken from HorizonNet (CVPR 2019)
    https://sunset1995.github.io/HorizonNet/
    """
    xs, ys = np.array(xs), np.array(ys)
    idx_sort = np.argsort(xs + ys / ys.max() * (int(y_small_first) * 2 - 1))
    xs, ys = xs[idx_sort], ys[idx_sort]
    _, idx_unique = np.unique(xs, return_index=True)
    xs, ys = xs[idx_unique], ys[idx_unique]
    assert np.all(np.diff(xs) > 0)
    return xs, ys


def cor_2_1d(cor, H, W):
    """
    Implementation taken from HorizonNet (CVPR 2019)
    https://sunset1995.github.io/HorizonNet/
    """
    bon_ceil_x, bon_ceil_y = [], []
    bon_floor_x, bon_floor_y = [], []
    n_cor = len(cor)
    for i in range(n_cor // 2):
        xys = panostretch.pano_connect_points(
            cor[i * 2], cor[(i * 2 + 2) % n_cor], z=-50, w=W, h=H
        )
        bon_ceil_x.extend(xys[:, 0])
        bon_ceil_y.extend(xys[:, 1])
    for i in range(n_cor // 2):
        xys = panostretch.pano_connect_points(
            cor[i * 2 + 1], cor[(i * 2 + 3) % n_cor], z=50, w=W, h=H
        )
        bon_floor_x.extend(xys[:, 0])
        bon_floor_y.extend(xys[:, 1])
    bon_ceil_x, bon_ceil_y = sort_xy_filter_unique(
        bon_ceil_x, bon_ceil_y, y_small_first=True
    )
    bon_floor_x, bon_floor_y = sort_xy_filter_unique(
        bon_floor_x, bon_floor_y, y_small_first=False
    )
    bon = np.zeros((2, W))
    bon[0] = np.interp(np.arange(W), bon_ceil_x, bon_ceil_y, period=W)
    bon[1] = np.interp(np.arange(W), bon_floor_x, bon_floor_y, period=W)
    bon = ((bon + 0.5) / H - 0.5) * np.pi
    return bon


def filter_out_noisy_layouts(list_ly, max_room_factor_size=2):
    """
    Filters out list_ly based on the cam2boundary defined for each ly. 
    All estimation which are greater than max_room_factor_size x mean(largest cam distances)
    Args:
        list_ly (_type_): List of Layout instances
        max_room_factor_size (int, optional): max ly size allowed. Defaults to 2.
    """
    # ! Filtering out noisy estimation
    logging.info(f"Filtering noisy layouts: initial #{list_ly.__len__()}")
    mean_ = np.median([ly.cam2boundary.max() for ly in list_ly])
    list_ly = [ly for ly in list_ly if ly.cam2boundary.max() < max_room_factor_size*mean_]
    logging.info(f"Filtering noisy layouts: final #{list_ly.__len__()}")


def normalize_list_ly(list_ly, scale_and_center=None):
    """
    Normalize the size of all layouts in a room between -1 to 1,
    centralizing all point to the median center of the room.
    """

    if scale_and_center is None:
        pcl = np.hstack([ly.boundary_floor for ly in list_ly])
        center = np.median(pcl, axis=1, keepdims=True)
        pcl = pcl - center
        scale = np.max(np.linalg.norm(pcl[(0, 2), :], axis=0))
    else: 
        scale, center = scale_and_center
    [ly.normalize_boundaries(scale, center) for ly in list_ly]
    return scale, center
    


def load_pseudo_labels(list_ly, hard_copy=False):
    cfg = list_ly[0].cfg
    room_scene = list_ly[0].cfg._room_scene

    if hard_copy:
        list_ly = [deepcopy(ly) for ly in list_ly]

    pseudo_labels_dir = cfg.mlc_labels[cfg.mlc_data].labels_dir
    list_fn = os.listdir(os.path.join(pseudo_labels_dir, 'mlc_label'))
    # Select only the pseudo labels which belong to the current room
    ps_labels = {
        Path(fn).stem: np.load(os.path.join(pseudo_labels_dir, 'mlc_label', fn))
        for fn in list_fn if room_scene in fn
    }
    [ly.recompute_data(phi_coord=ps_labels[ly.idx]) for ly in list_ly]
    return list_ly
