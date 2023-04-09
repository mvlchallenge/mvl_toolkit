import argparse
from mvl_challenge import (
    ASSETS_DIR,
    EPILOG,
    DEFAULT_MVL_DIR,
)
from mvl_challenge.utils.io_utils import (
    get_files_given_a_pattern,
    get_scene_room_from_scene_room_idx,
)
import os
import yaml
import json
from pathlib import Path
from mvl_challenge.utils.io_utils import (
    save_json_dict,
    create_directory,
    get_scene_list_from_dir,
)
from mvl_challenge.config.cfg import get_empty_cfg
from mvl_challenge.scene_list__edit_info import prune_list_frames
from mvl_challenge.config.cfg import set_loggings
import numpy as np
from tqdm import tqdm


def fix_scene_list_file(args):
    
    assert os.path.exists(args.scene_list), f"No directory found {args.scene_list}"
    
    data = json.load(open(args.scene_list, 'r'))
    list_scenes = [scene for scene in data.values()]
    list_scenes = [item for sublist in list_scenes for item in sublist]

    list_rooms = np.unique(
        [get_scene_room_from_scene_room_idx(Path(fn).stem) for fn in list_scenes]
    ).tolist()
    data_new = {}
    for room in tqdm(list_rooms, desc="List rooms..."):
        data_new[room] = [Path(fn).stem for fn in list_scenes if f"{room}_" in fn]

    list_rooms = list(data.keys())
    check_rooms = [room not in list_rooms for room in data_new.keys()]
    if np.sum(check_rooms) > 0:
        if args.c:
            print_difference(data, data_new)
        else:
            print_difference(data, data_new)
            save_json_dict(f'{args.scene_list}', data_new)
    else:
        print("Nothing to change...")  
    
def print_difference(data, data_new):
    list_rooms = list(data.keys())
    print("New Rooms found:")
    [print(f"\t{room}") for room in data_new.keys() if room not in list_rooms ]
    
    
    
def get_argparse():
    parser = argparse.ArgumentParser(epilog=EPILOG)

    parser.add_argument(
        "-f",
        "--scene_list",
        # required=True,
        default="/media/NFS/kike/360_Challenge/mvl_toolkit/mvl_challenge/data/scene_list/scene_list__warm_up_training.json",
        type=str,
        help="*.json scene list file",
    )


    parser.add_argument(
        "-c",
        action='store_true',
        help="Overwrite passed scene list file",
    )
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_argparse()
    fix_scene_list_file(args)
