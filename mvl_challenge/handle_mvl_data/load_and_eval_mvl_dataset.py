import argparse
from mvl_challenge import DATA_DIR, ROOT_DIR, CFG_DIR, EPILOG, ASSETS_DIR
from mvl_challenge.config.cfg import read_omega_cfg
from mvl_challenge.datasets.mvl_dataset import MVLDataset, iter_mvl_room_scenes
import logging
from tqdm import tqdm
from mvl_challenge.utils.vispy_utils import plot_list_ly
from mvl_challenge.utils.image_utils import draw_boundaries_phi_coord
from imageio import imwrite
from mvl_challenge.models.wrapper_horizon_net import WrapperHorizonNet
import numpy as np
from mvl_challenge.utils.io_utils import create_directory, get_scene_room_from_scene_room_idx
from mvl_challenge.utils.image_utils import plot_image

def get_cfg_from_args(args):
    cfg = read_omega_cfg(args.cfg)
    cfg.scene_dir = args.scene_dir
    cfg.scene_list = args.scene_list
    cfg.ckpt = args.ckpt
    cfg.cuda = args.cuda
    return cfg


def main(args):
    cfg = get_cfg_from_args(args)
    mvl = MVLDataset(cfg)
    hn = WrapperHorizonNet(cfg)

    for list_ly in iter_mvl_room_scenes(model=hn, dataset=mvl):
        for ly in list_ly:
            img = ly.get_rgb()
            draw_boundaries_phi_coord(img, phi_coord=ly.phi_coord)
            plot_image(
                image=img, 
                caption=ly.idx
            )
        plot_list_ly(list_ly)
        

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
        help='MVL dataset directory.'
    )

    parser.add_argument(
        "--cfg",
        type=str,
        default=f"{CFG_DIR}/eval_mvl_dataset.yaml",
        help="Config file to load a MVL dataset."
    )

    parser.add_argument(
        "-f", "--scene_list",
        type=str,
        help="Config file to load a MVL dataset."
    )

    parser.add_argument(
        "--ckpt",
        default="mp3d",
        help="Pretrained model ckpt."
    )

    parser.add_argument(
        "--cuda",
        default=0,
        type=int,
        help="Cuda device."
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_argparse()
    main(args)
