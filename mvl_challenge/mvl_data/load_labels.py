import argparse
from mvl_challenge import DATA_DIR, ROOT_DIR, CFG_DIR, EPILOG, ASSETS_DIR
from mvl_challenge.config.cfg import read_omega_cfg
from mvl_challenge.datasets.mvl_dataset import MVLDataset, iter_mvl_room_scenes
import logging
import os
from tqdm import tqdm
from mvl_challenge.utils.vispy_utils import plot_list_ly
from mvl_challenge.utils.image_utils import draw_boundaries_phi_coords
from imageio import imwrite
from mvl_challenge.models.wrapper_horizon_net import WrapperHorizonNet
import numpy as np
from mvl_challenge.utils.io_utils import create_directory, get_scene_room_from_scene_room_idx, load_gt_label
from mvl_challenge.utils.image_utils import plot_image


def get_cfg_from_args(args):
    cfg = read_omega_cfg(args.cfg)
    cfg.scene_dir = args.scene_dir
    cfg.scene_list = args.scene_list
    return cfg



def main(args):
    cfg = get_cfg_from_args(args)
    mvl = MVLDataset(cfg)
    
    for list_ly in mvl.iter_list_ly():
        for ly in list_ly:
            phi_coords = load_gt_label(os.path.join(
                args.scene_dir, "labels", "gt", f"{ly.idx}.npz"
            )) 
            img = ly.get_rgb()
            draw_boundaries_phi_coords(
                img,
                phi_coords=phi_coords
                )
            
            plot_image(
                image=img,
                caption=ly.idx + " - GT labels"
            )


def get_argparse():
    desc = "This script loads a MVL dataset given a passed scene directory, scene list and cfg file. " + \
        "The scene directory is where the MVL data is stored. " + \
        "The scene list is the list of scene in scene_room_idx format. " + \
        "The cfg file is the yaml configuration with all hyperparameters set to default values."

    parser = argparse.ArgumentParser(
        description=desc,
        epilog=EPILOG
    )

    parser.add_argument(
        '-d', '--scene_dir',
        type=str,
        # default=f'{ASSETS_DIR}/mvl_data/mp3d_fpe',
        help='MVL dataset directory.'
    )

    parser.add_argument(
        "--cfg",
        type=str,
        default=f"{CFG_DIR}/load_mvl_dataset.yaml",
        help=f"Config file to load a MVL dataset. (Default: {CFG_DIR}/load_mvl_dataset.yaml)"
    )

    parser.add_argument(
        "-f", "--scene_list",
        type=str,
        # default=f"{DATA_DIR}/mp3d_fpe/test__gt_labels__scene_list.json",
        help="Config file to load a MVL dataset."
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_argparse()
    main(args)
