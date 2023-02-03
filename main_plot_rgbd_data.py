import argparse

import numpy as np

from mvl_datasets.datasets.rgbd_datasets import HM3D_MVL, MP3D_FPE
from mvl_datasets.utils.vispy_utils import plot_color_plc


def main(args):
    
    if args.dataset_name == "mp3d_fpe":
        dt = MP3D_FPE.from_args(args)
    elif args.dataset_name == "hm3d_mvl":
        dt = HM3D_MVL.from_args(args)
    else:
        raise ValueError()
    
    list_fr = dt.get_list_frames()
    pcl  = np.hstack([fr.get_pcl() for fr in list_fr[:4]])
    plot_color_plc(points=pcl[0:3, :].T, color=pcl[3:].T)

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
    
    parser.add_argument(
        '--dataset_name',
        # required=True,
        default="hm3d_mvl",
        type=str,
        help='dataset name [mp3d_fpe or hmd3d_mvl]'
    )

    args = parser.parse_args()
    return args
    
if __name__ == '__main__':
    args = get_args()
    main(args)
    