import argparse
from mvl_challenge import ASSETS_DIR, EPILOG
from mvl_challenge.utils.io_utils import (
    get_files_given_a_pattern,
    get_scene_room_from_scene_room_idx,
)
import os
import json
from mvl_challenge.utils.io_utils import (
    save_json_dict,
    create_directory,
    get_scene_list_from_dir,
)
from mvl_challenge.config.cfg import get_empty_cfg
import numpy as np
from tqdm import tqdm


def print_scene_data_info(scene_data):
    frames_list = [fr for fr in scene_data.values()]
    numb_frames = [fr.__len__() for fr in scene_data.values()]
    frames_list = [item for sublist in frames_list for item in sublist]
    print(f"Number Rooms: {list(scene_data.keys()).__len__()}")
    print(f"Number Frames: {frames_list.__len__()}")
    if frames_list.__len__() == 0:
        return
    print(f"Max Number Frames: {np.max(numb_frames)}")
    print(f"Min Number Frames: {np.min(numb_frames)}")
    print(f"Average Number Frames: {np.mean(numb_frames)}")


def print_stats_from_scene_list(args):
    cfg = get_empty_cfg()
    if os.path.isdir(args.i):
        cfg.scene_dir = args.i
        scene_data = get_scene_list_from_dir(cfg)

    elif os.path.isfile(args.i):
        cfg.scene_list = args.i
        scene_data = json.load(open(cfg.scene_list))
    else:
        raise ValueError("Wrong input argument")

    print_scene_data_info(scene_data)

    if args.output_fn != "":
        create_directory(os.path.dirname(args.output_fn), delete_prev=False)
        save_json_dict(args.output_fn, scene_data)
        print(f"Saved at {args.output_fn}")


def get_argparse():
    desc = "This script reads a scene_list.json file or a mvl directory to print the pyth    number of frames and rooms in the data."

    parser = argparse.ArgumentParser(description=desc, epilog=EPILOG)

    parser.add_argument(
        "--i", required=True, type=str, help="Input Filename or directory."
    )

    parser.add_argument(
        "-o",
        "--output_fn",
        default="",
        type=str,
        help="Outputs the scene_list information in the passed filename.",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_argparse()
    print_stats_from_scene_list(args)
