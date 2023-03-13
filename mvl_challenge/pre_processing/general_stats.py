import argparse
from mvl_challenge import ASSETS_DIR, EPILOG
from mvl_challenge.utils.io_utils import get_files_given_a_pattern, get_scene_room_from_scene_room_idx
import os
import yaml
import json
from pathlib import Path
from mvl_challenge.utils.io_utils import save_json_dict, create_directory
from mvl_challenge.config.cfg import set_loggings
import numpy as np
from tqdm import tqdm


def data_from_scene_list(args):
    scene_data = json.load(open(args.scene_list))
    print(f"Number Rooms: {list(scene_data.keys()).__len__()}")
    frames_list = [fr for fr in scene_data.values()]
    frames_list = [item for sublist in frames_list for item in sublist]
    print(f"Number Frames: {frames_list.__len__()}")
    
def based_on_scene_list(args):
    scene_data = json.load(open(args.scene_list))
    split_list = json.load(open(args.split_list))
    
    for split, scenes in split_list.items():
        print(f"Split: {split}")
        # rooms_list  = [r for r in list(scene_data.keys()) if r.split("_room")[0] in scenes]
        rooms_list  = [r for r in list(scene_data.keys()) if r.split("_")[0] in scenes]
        frames_list = [fr for r, fr in scene_data.items() if r in rooms_list]
        frames_list = [item for sublist in frames_list for item in sublist]

        print(f"Number Rooms: {rooms_list.__len__()}")
        print(f"Number Frames: {frames_list.__len__()}")


def get_argparse():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-f', '--scene_list',
        # required=True,
        default=f"{ASSETS_DIR}/stats/hm3d_mvl__train__scene_list.json",
        # default=f"{ASSETS_DIR}/stats/mp3d_fpe__multiple_rooms__scene_list.json",
        # default="/media/NFS/kike/360_Challenge/mvl_toolkit/mvl_challenge/data/mp3d_fpe/test__gt_labels__scene_list.json",
        type=str,
        help='Filename to the scene list (scene_room_idx json file).'
    )
    
    parser.add_argument(
        '-s', '--split_list',
        # required=True,
        default=f"{ASSETS_DIR}/stats/scene_splits__hm3d_mvl.json",
        type=str,
        help='split list.'
    )
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_argparse()
    # data_from_scene_list(args)
    based_on_scene_list(args)