import argparse

import numpy as np

import matplotlib.pyplot as plt

import logging
from mvl_datasets.datasets.rgbd_datasets import RGBD_Dataset
from mvl_datasets.utils.vispy_utils import plot_color_plc
from mvl_datasets.utils.geometry_utils import extend_array_to_homogeneous
from mvl_datasets import CFG_DIR, ASSETS_DIR
import pyransac3d as pyrsc
from tqdm import tqdm
from mvl_datasets.config.cfg import read_omega_cfg


def get_masked_pcl(cfg, list_fr):

    # Skipped the first camera frame to avoid noise
    pcl = np.hstack([fr.get_pcl() for fr in list_fr[1:10]])
    color = pcl[3:, :]
    pcl = np.linalg.inv(list_fr[0].pose)[:3, :] @ extend_array_to_homogeneous(pcl[:3, :])
    # masking around the first camera frame
    mask = np.linalg.norm(pcl[(0, 2), :], axis=0) < cfg.xz_radius
    pcl = pcl[:, mask]
    color = color[:, mask]
    # Masking the floor
    mask = pcl[1, :] > cfg.min_height
    pcl = pcl[:, mask]
    color = color[:, mask]
    # plot_color_plc(points=pcl[0:3, :].T, color=color.T)

    idx = np.linspace(0, pcl.shape[1]-1,  pcl.shape[1]).astype(np.int32)
    np.random.shuffle(idx)
    return pcl[:3, idx[:cfg.min_samples]], color[:, idx[:cfg.min_samples]]


def estimate_camera_height(cfg, list_fr):
    cam_h_hyp = []
    # for _ in tqdm(range(cfg.iter), desc="Iter camera Height..."):
    iter_progress = tqdm(cfg.iter, desc="Iter camera Height...")
    iteration = 0
    while True:
        np.random.shuffle(list_fr)
        logging.info(f"Number fr: {list_fr.__len__()}")
        pcl, color = get_masked_pcl(cfg, list_fr)
        if pcl.size == 0:
            continue
        pln = pyrsc.Plane()
        best_eq, best_inliers = pln.fit(pcl.T, cfg.fit_error)
        # plot_color_plc(points=pcl[0:3, best_inliers].T, color=color[:, best_inliers].T)

        # ! since each camera fr does not have the same height
        cam_h2room = np.abs(best_eq[-1]) + list_fr[0].pose[1, 3]
        cam_h_hyp.append(cam_h2room)
        iteration += 1
        iter_progress.update(iteration)

        if cfg.iter == iteration:
            break
    return np.median(cam_h_hyp)


def main(args):
    dt = RGBD_Dataset.from_args(args)
    list_fr = dt.get_list_frames()
    # ! cfg file for camera height estimation
    cfg_cam = read_omega_cfg(args.cfg)
    h = estimate_camera_height(cfg_cam, list_fr)
    logging.info(f"Estimated camera height: {h}")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--scene_dir',
        # required=True,
        default="/media/public_dataset/MP3D_360_FPE/SINGLE_ROOM_SCENES/2t7WUuJeko7/0/",
        type=str,
        help='Directory of all scene in the dataset'
    )

    parser.add_argument(
        '--cfg',
        default=f"{CFG_DIR}/camera_height.yaml",
        help='Cfg tp compute camera height'
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    main(args)
