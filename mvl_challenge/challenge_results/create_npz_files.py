import argparse
from mvl_challenge import DATA_DIR, ROOT_DIR, CFG_DIR, EPILOG, ASSETS_DIR
from mvl_challenge.config.cfg import read_omega_cfg
from mvl_challenge.datasets.mvl_dataset import MVLDataset, iter_mvl_room_scenes
from mvl_challenge.utils.vispy_utils import plot_list_ly
from mvl_challenge.utils.image_utils import draw_boundaries_phi_coords
from imageio import imwrite
from mvl_challenge.models.wrapper_horizon_net import WrapperHorizonNet
from mvl_challenge.utils.io_utils import create_directory, get_scene_room_from_scene_room_idx, save_compressed_phi_coords
from mvl_challenge.utils.image_utils import plot_image
import numpy as np
import os
from pathlib import Path



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

    # ! Join the output_dir and the scene_list
    output_dir = create_directory(os.path.join(args.output_dir, Path(args.scene_list).stem), delete_prev=False)
    for list_ly in iter_mvl_room_scenes(model=hn, dataset=mvl):
        for ly in list_ly:            
            fn = os.path.join(output_dir, ly.idx)
            # ! IMPORTANT: Use ALWAYS save_compressed_phi_coords()
            save_compressed_phi_coords(ly.phi_coords, fn)
            

def get_argparse():
    desc = "This script saves the evaluations (phi_coords) of HorizonNet given a MVL dataset, scene list and cfg file. " + \
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
        default=f"{CFG_DIR}/eval_mvl_dataset.yaml",
        help=f"Config file to load a MVL dataset. For this script model cfg most be defined in the cfg file too. (Default: {CFG_DIR}/eval_mvl_dataset.yaml)"
    )

    parser.add_argument(
        "-f", "--scene_list",
        type=str,
        # default=f"{DATA_DIR}/mp3d_fpe/mp3d_fpe__test__scene_list.json",
        help="Scene_list of mvl scenes in scene_room_idx format."
    )

    parser.add_argument(
        "--ckpt",
        default="mp3d",
        help="Pretrained model ckpt (Default: mp3d)"
    )

    parser.add_argument(
        "--cuda",
        default=0,
        type=int,
        help="Cuda device. (Default: 0)"
    )

    parser.add_argument(
        "-o", "--output_dir",
        required=True,
        help="Output directory where to store phi_coords estimations."
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_argparse()
    main(args)
