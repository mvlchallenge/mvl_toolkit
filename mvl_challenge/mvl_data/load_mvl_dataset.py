import argparse
from mvl_challenge import (
    DATA_DIR,
    ROOT_DIR,
    CFG_DIR,
    EPILOG,
    ASSETS_DIR,
    DEFAULT_MVL_DIR,
    SCENE_LIST_DIR,
)
from mvl_challenge.config.cfg import read_omega_cfg
from mvl_challenge.datasets.mvl_dataset import MVLDataset
import logging
from tqdm import tqdm
import numpy as np
from mvl_challenge.utils.image_utils import plot_image, add_caption_to_image


def get_cfg_from_args(args):
    cfg = read_omega_cfg(args.cfg)
    cfg.scene_dir = args.scene_dir
    cfg.scene_list = args.scene_list
    return cfg


def main(args):
    cfg = get_cfg_from_args(args)
    mvl = MVLDataset(cfg)
    mvl.print_mvl_data_info()
    # ! Loading list_ly by passing room_scene
    for room_scene in tqdm(mvl.list_rooms, desc="Loading room scene..."):
        list_ly = mvl.get_list_ly(room_scene=room_scene)
        for ly in list_ly:
            plot_image(image=ly.get_rgb(), caption=ly.idx)

    # ! Iterator of list_ly
    for list_ly in mvl.iter_list_ly():
        for ly in list_ly:
            plot_image(image=ly.get_rgb(), caption=ly.idx)


def get_argparse():
    desc = (
        "This script loads a MVL dataset given a passed scene directory, scene list and cfg file. "
        + "The scene directory is where the MVL data is stored. "
        + "The scene list is the list of scene in scene_room_idx format. "
        + "The cfg file is the yaml configuration with all hyperparameters set to default values."
    )

    parser = argparse.ArgumentParser(description=desc, epilog=EPILOG)

    parser.add_argument(
        "-d",
        "--scene_dir",
        # required=True,
        # default=f"{ASSETS_DIR}/mvl_data/mp3d_fpe",
        # default=f'{ASSETS_DIR}/tmp/zip_files',
        # default=None,
        default=f"{DEFAULT_MVL_DIR}",
        type=str,
        help="MVL dataset directory.",
    )

    parser.add_argument(
        "--cfg",
        type=str,
        default=f"{CFG_DIR}/load_mvl_dataset.yaml",
        help=f"Config file to load a MVL dataset. (default: {CFG_DIR}/load_mvl_dataset.yaml)",
    )

    parser.add_argument(
        "-f",
        "--scene_list",
        type=str,
        default=f"{SCENE_LIST_DIR}/scene_list__warm_up_pilot_set.json",
        help="Config file to load a MVL dataset.",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_argparse()
    main(args)
