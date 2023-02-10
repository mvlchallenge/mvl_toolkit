import argparse

import numpy as np

import matplotlib.pyplot as plt

import logging
from mvl_datasets.datasets.rgbd_datasets import RGBD_Dataset
from mvl_datasets.utils.vispy_utils import plot_color_plc
from mvl_datasets.utils.geometry_utils import extend_array_to_homogeneous
import pyransac3d as pyrsc
from tqdm import tqdm

def get_masked_pcl(args, list_fr):
    
    # Skipped the first camera frame to avoid noise
    pcl  = np.hstack([fr.get_pcl() for fr in list_fr[1:10]])
    color = pcl[3:, :]
    pcl = np.linalg.inv(list_fr[0].pose)[:3, :] @ extend_array_to_homogeneous(pcl[:3, :])
    # masking around the first camera frame
    mask  = np.linalg.norm(pcl[(0, 2), :], axis=0) < args.xz_distance
    pcl = pcl[:, mask]
    color = color[:, mask]
    # Masking the floor
    mask  = pcl[1, :] > args.min_height
    pcl = pcl[:, mask]
    color = color[:, mask]
    # plot_color_plc(points=pcl[0:3, :].T, color=color.T) 
    
    idx = np.linspace(0, pcl.shape[1]-1,  pcl.shape[1]).astype(np.int32)
    np.random.shuffle(idx)
    return pcl[:3, idx[:args.min_samples]], color[:, idx[:args.min_samples]]

def estimate_camera_height(args, list_fr):
    cam_h_hyp = []

    for _ in tqdm(range(args.iter), desc="Iter camera Height...`"):
        np.random.shuffle(list_fr)
        pcl, color= get_masked_pcl(args, list_fr)
        pln = pyrsc.Plane()
        best_eq, best_inliers = pln.fit(pcl.T, args.fit_error)
        # plot_color_plc(points=pcl[0:3, best_inliers].T, color=color[:, best_inliers].T) 

        # ! since each camera fr does not have the same height
        cam_h2room =  np.abs(best_eq[-1]) + list_fr[0].pose[1, 3]
        cam_h_hyp.append(cam_h2room)
        
    return np.median(cam_h_hyp)
    
def main(args):  
    dt = RGBD_Dataset.from_args(args)
    list_fr = dt.get_list_frames()
    h = estimate_camera_height(args, list_fr)
    logging.info(f"Estimated camera height: {h}")
    
def get_args(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    # * Input Directory (-s)
    parser.add_argument(
        '--scene_dir',
        # required=True,
        default="/media/public_dataset/HM3D-MVL/test/BHXhpBwSMLh/0/",
        type=str,
        help='Input Directory (scene directory defined until version scene)'
    )
    
    parser.add_argument(
        '--fit_error',
        # required=True,
        default=0.001,
        help='How much variance is allowed for RANSAC plane estimation'
    )
    
    parser.add_argument(
        '--xz_distance',
        # required=True,
        default=1,
        help='Max distance in XZ around the 1st camera frame'
    )
    
    parser.add_argument(
        '--min_height',
        # required=True,
        default=0.8,
        help='Min camera height from camera frames to floor'
    )
    
    
    parser.add_argument(
        '--min_samples',
        # required=True,
        default=1000,
        help='Min number of point used for RANSAC plane estimation'
    )
    
    args = parser.parse_args()
    return args
    
if __name__ == '__main__':
    args = get_args()
    main(args)
    