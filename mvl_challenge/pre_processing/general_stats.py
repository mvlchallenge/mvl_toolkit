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


def print_stats_from_scene_list(args):
    scene_data = json.load(open(args.scene_list))
    frames_list = [fr for fr in scene_data.values()]
    numb_frames = [fr.__len__() for fr in scene_data.values()]
    frames_list = [item for sublist in frames_list for item in sublist]
    print(f"Number Rooms: {list(scene_data.keys()).__len__()}")
    print(f"Number Frames: {frames_list.__len__()}")
    print(f"Max Number Frames: {np.max(numb_frames)}")
    print(f"Min Number Frames: {np.min(numb_frames)}")
    print(f"Average Number Frames: {np.mean(numb_frames)}")

    
def get_argparse():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-f', '--scene_list',
        # required=True,
        # default=f"{ASSETS_DIR}/stats/hm3d_mvl__train__scene_list.json",
        # default=f"{ASSETS_DIR}/stats/mp3d_fpe__multiple_rooms__scene_list.json",
        # default="/media/public_dataset/mvl_challenge/hm3d_mvl/03.14.2023__all_hm3d_mvl__scene_list.json",
        default="/media/public_dataset/mvl_challenge/hm3d_mvl/scene_list__test.json",
        type=str,
        help='Filename to the scene list (scene_room_idx json file).'
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_argparse()
    print_stats_from_scene_list(args)
