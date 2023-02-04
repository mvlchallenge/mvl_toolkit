import argparse

import numpy as np

import matplotlib.pyplot as plt

import logging
from mvl_datasets.datasets.rgbd_datasets import RGBD_Dataset
from mvl_datasets.utils.vispy_utils import plot_color_plc

def estimate_camera_height(list_fr):
    pcl  = np.hstack([fr.get_pcl() for fr in list_fr[:10]])
    # masking around the first camera frame
    mask  = np.linalg.norm(pcl[(0, 2), :], axis=0) < 1
    pcl = pcl[:, mask]
    # plot_color_plc(points=pcl[0:3, :].T, color=pcl[3:].T)
    mask  = pcl[1, :] > np.abs(np.max(pcl[1, :]) - 0.1)
    pcl = pcl[:, mask]
    # plot_color_plc(points=pcl[0:3, :].T, color=pcl[3:].T)  
    h =  np.mean(abs((pcl[1, :] - np.max(pcl[1, :]))))
    return 

def main(args):  
    dt = RGBD_Dataset.from_args(args)
    list_fr = dt.get_list_frames()
    h = estimate_camera_height(list_fr)
    logging.info(f"Estimated camera height:{h}")
    
def get_args():
    parser = argparse.ArgumentParser()
    # * Input Directory (-s)
    parser.add_argument(
        '--scene_dir',
        # required=True,
        default="/media/public_dataset/HM3D-MVL/test/BHXhpBwSMLh/0/",
        type=str,
        help='Input Directory (scene directory defined until version scene)'
    )
    args = parser.parse_args()
    return args
    
if __name__ == '__main__':
    args = get_args()
    main(args)
    