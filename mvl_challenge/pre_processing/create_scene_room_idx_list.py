import argparse
from mvl_challenge import ASSETS_DIR, EPILOG
from mvl_challenge.utils.io_utils import get_files_given_a_pattern, get_scene_room_from_scene_room_idx
import os
import yaml
import json
from pathlib import Path
from mvl_challenge.utils.io_utils import save_json_dict, create_directory, get_scene_list_from_dir
from mvl_challenge.pre_processing.pre_process_scene_list import prune_list_frames
from mvl_challenge.config.cfg import set_loggings
import numpy as np
from tqdm import tqdm


def get_list_idx_from_dir(dir):
    """
    Returns the list of frames defined in the passed directory

    Args:
        dir (dirname): 

    Returns:
        list: list_idx
    """
    return [int(Path(idx).stem) for idx in os.listdir(dir)]


def get_list_rooms_idx_from_metadata(metadata_filename):
    """     
    Returns the list of rooms and index frames defined for the passed metadata file
    Args:
        metadata_file (string): metadata filename
    Returns:
        list: [room_idx]
    """
    data = yaml.safe_load(open(metadata_filename))
    list_rooms = [r for r in list(data.keys()) if "room" in r]
    list_room_idx = []
    for room in list_rooms:
        list_kf = data[room]['list_kf']
        [list_room_idx.append(f"{room.replace('.', '')}_{kf}")
         for kf in list_kf]
    return list_room_idx


def get_list_rooms_idx_from_mvl_labels(mvl_labels_filename):
    """
    Returns the list of room_idx defined by mvl_labels.

    Args:
        mvl_labels_filename (string): mvl_labels filename
    Returns:
        [list room_idx]: List of room_idx
    """

    mvl_data = json.load(open(mvl_labels_filename, "r"))
    list_room_idx = []
    for room, list_kf in enumerate(mvl_data['list_kf']):
        [
            list_room_idx.append(f"room{room}_{kf}")
            for kf in list_kf
        ]
    return list_room_idx


def get_list_scene_room_idx(args):
    list_scenes = get_files_given_a_pattern(
        data_dir=args.scene_dir,
        flag_file="frm_ref.txt",
        exclude=['rgb', 'depth'],
    )
    assert list_scenes.__len__() > 0, f"No scenes were found at {args.scene_dir}"

    print(f"Total Number of scenes found: {list_scenes.__len__()}")
    # ! Construct list scene_version string
    list_scene_version_room_frames = []
    for scene in tqdm(list_scenes, desc="List scenes..."):
        scene_version = scene.split("/")

        if scene_version[-1] == '':
            scene_version.pop()

        filename = "_".join(scene.split("/")[-2:])

        if os.path.exists(os.path.join(scene, "mvl_challenge_labels.json")):
            # ! If mvl_challenge_labels exists for this scene
            mvl_labels_fn = os.path.join(scene, "mvl_challenge_labels.json")
            list_room_idx = get_list_rooms_idx_from_mvl_labels(mvl_labels_fn)
        # ! Check multi-room scenes
        elif os.path.exists(os.path.join(scene, "metadata")):
            # ! If metadata exists
            metadata_filename = os.path.join(scene, "metadata", "room_gt_v0.0.yaml")
            list_room_idx = get_list_rooms_idx_from_metadata(metadata_filename)
        else:
            # continue
            # ! defining room_scene_idx from rgb images directly
            list_idx = get_list_idx_from_dir(os.path.join(scene, "rgb"))
            list_room_idx = [f"room0_{idx}" for idx in list_idx]

        [list_scene_version_room_frames.append(f"{filename}_{room_idx}")
         for room_idx in list_room_idx
         ]

    return list_scene_version_room_frames


def save_scene_list_from_mvl_directory(args):
    assert os.path.exists(args.scene_dir), f"No directory found {args.scene_dir}"

    set_loggings()
    data_dict = get_scene_list_from_dir(args)

    create_directory(args.output_dir, delete_prev=False)
    fn = os.path.join(args.output_dir, Path(f"{args.output_filename}").stem)
    save_json_dict(f"{fn}.json", data_dict)



def scene_list_from_rgbd_dataset(args):
    list_scene_room_idx = get_list_scene_room_idx(args)
    print(f"Total number of frames found: {list_scene_room_idx.__len__()}")

    list_rooms = np.unique([
        get_scene_room_from_scene_room_idx(r)
        for r in list_scene_room_idx
    ]).tolist()

    print(f"Total number of rooms found: {list_rooms.__len__()}")
    data_dict = {}
    for room in tqdm(list_rooms, desc="List rooms..."):
        list_frames = [fr for fr in list_scene_room_idx if f"{room}_" in fr]
        if args.max_fr is not None:
            prune_list_frames(list_frames, args.max_fr)
        if list_frames.__len__() > args.min_fr:
            data_dict[room] = list_frames

    create_directory(args.output_dir, delete_prev=False)
    fn = os.path.join(args.output_dir, Path(f"{args.output_filename}").stem)
    save_json_dict(f"{fn}.json", data_dict)


def get_argparse():
    desc = "This script reads a RGBD dataset directory to create and saves a `scene_room_idx` file. " + \
        "A 'scene_room_idx' file is a list of all frames encoded as scene+room+idx, e.g., 2t7WUuJeko7_1_room0_2606"

    parser = argparse.ArgumentParser(
        description=desc,
        epilog=EPILOG
    )

    parser.add_argument(
        '-d', '--scene_dir',
        # required=True,
        # default=None,
        # default="/media/public_dataset/MP3D_360_FPE/SINGLE_ROOM_SCENES/",
        default="/media/public_dataset/HM3D-MVL/train",
        type=str,
        help='RGBD dataset directory.'
    )

    parser.add_argument(
        '-f', '--output_filename',
        # required=True,
        default="scene_room_frames.json",
        type=str,
        help='Filename to the scene_room_idx file.'
    )

    parser.add_argument(
        '-o', '--output_dir',
        # required=True,
        default=f"{ASSETS_DIR}/tmp/test",
        type=str,
        help=f'Output directory for the output_file to be created. (Default: "{ASSETS_DIR}/tmp")'
    )

    parser.add_argument(
        '-m', '--min_fr',
        # required=True,
        default=5,
        type=int,
        help='Minimum number of frames per room. (Default: 5)'
    )

    parser.add_argument(
        '-x', '--mvl_dir',
        # required=True,
        # default=f"{ASSETS_DIR}/mvl_data/geometry_info",
        # default=None,
        action='store_true',
        help='is the passed scene_dir a MVL directory? (data stored in  scene_room_idx format). Default: False'
    )

    parser.add_argument(
        '--max_fr',
        # required=True,
        # default=f"{ASSETS_DIR}/mvl_data/geometry_info",
        default=50,
        help='MVL directory of files saved in scene_room_idx format. (Default: 50)'
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_argparse()
    if not args.mvl_dir:
        scene_list_from_rgbd_dataset(args)
    else:
        save_scene_list_from_mvl_directory(args)
