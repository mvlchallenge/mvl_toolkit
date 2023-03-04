import argparse
from mvl_challenge import ASSETS_DIR, EPILOG
from mvl_challenge.utils.io_utils import get_files_given_a_pattern
import os
import yaml
from pathlib import Path
from mvl_challenge.utils.io_utils import get_scene_room_from_scene_room_idx
from mvl_challenge.utils.io_utils import save_json_dict, create_directory
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

        # ! Check multi-room scenes
        if os.path.exists(os.path.join(scene, "metadata")):
            metadata_filename = os.path.join(scene, "metadata", "room_gt_v0.0.yaml")
            list_room_idx = get_list_rooms_idx_from_metadata(metadata_filename)
        else:
            list_idx = get_list_idx_from_dir(os.path.join(scene, "rgb"))
            list_room_idx = [f"room0_{idx}" for idx in list_idx]
        [list_scene_version_room_frames.append(f"{filename}_{room_idx}")
         for room_idx in list_room_idx
         ]

    return list_scene_version_room_frames


def main(args):
    list_scene_room_idx = get_list_scene_room_idx(args)
    print(f"Total number of frames found: {list_scene_room_idx.__len__()}")
    
    list_rooms = np.unique([
        get_scene_room_from_scene_room_idx(r)
        for r in list_scene_room_idx
    ]).tolist()

    print(f"Total number of rooms found: {list_rooms.__len__()}")
    data_dict = {}
    for room in tqdm(list_rooms, desc="List rooms..."):
        list_frames = [fr for fr in list_scene_room_idx if room in fr]
        data_dict[room] = list_frames

    create_directory(args.output_dir, delete_prev=False)
    fn = os.path.join(args.output_dir, Path(f"{args.output_filename}").stem)
    save_json_dict(f"{fn}.json", data_dict)


def get_argparse():
    desc = "This script reads a MV data directory to create and saves a `scene_room_idx` file. " + \
        "A 'scene_room_idx' file is a list of all frames encoded as scene+room+idx, e.g., 2t7WUuJeko7_1_room0_2606"

    parser = argparse.ArgumentParser(
        description=desc,
        epilog=EPILOG
    )

    parser.add_argument(
        '-d', '--scene_dir',
        # required=True,
        default="/media/public_dataset/MP3D_360_FPE/MULTI_ROOM_SCENES/",
        # default="/media/public_dataset/HM3D-MVL/test/BHXhpBwSMLh",
        type=str,
        help='MV data scene directory.)'
    )

    parser.add_argument(
        '-f', '--output_filename',
        # required=True,
        default="scene_room_frames.json",
        type=str,
        help='File name for the sene_room_frame file (default: scene_room_frames.json)'
    )

    parser.add_argument(
        '-o', '--output_dir',
        # required=True,
        default=f"{ASSETS_DIR}/mvl_data",
        type=str,
        help='Output directory for the output_file to be created.'
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_argparse()
    main(args)
