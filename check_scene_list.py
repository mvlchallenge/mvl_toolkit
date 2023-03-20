import argparse
import os
import json
import numpy as np
from mvl_challenge import EPILOG, ASSETS_DIR, DEFAULT_MVL_DIR, SCENE_LIST_DIR


def main(args):
    data = json.load(open(args.scene_list, "r"))
    list_frames = [f for f in data.values()]
    list_frames = [item for sublist in list_frames for item in sublist]
    
    list_imgs = [os.path.join(args.data_dir, 'img', f"{fn}.jpg") for fn in list_frames]
    list_geom_info = [os.path.join(args.data_dir, 'geometry_info', f"{fn}.json") for fn in list_frames]
    list_labels = [os.path.join(args.data_dir, 'labels', 'gt', f"{fn}.npz") for fn in list_frames]
    
    print(f" - Scene list: {args.scene_list}")
    print(f"* Total rooms: {list(data.keys()).__len__()}")
    print(f"* Total frames: {list_frames.__len__()}")
    if np.sum([os.path.isfile(fn) for fn in list_imgs]) == list_imgs.__len__():
        print(f" - [PASSED]\tAll images were found")
    else:
        if not args.v:
            print(f" - [FAILED]\tNot all images were found")
        else:
            [print(f" - [FAILED]\tNot found {fn}") for fn in list_imgs if not os.path.exists(fn)]
      
    if np.sum([os.path.isfile(fn) for fn in list_geom_info]) == list_geom_info.__len__():
        print(f" - [PASSED]\tAll JSON geometry files were found")
    else:
        if not args.v:
            print(f" - [FAILED]\tNot all JSON geometry files were found")
        else:
            [print(f" - [FAILED]\tNot found {fn}") for fn in list_geom_info if not os.path.exists(fn)]
      
    if np.sum([os.path.isfile(fn) for fn in list_labels]) == list_labels.__len__():
        print(f" - [PASSED]\tAll labels *.npz files were found")
    else:
        if not args.v:
            print(f" - [FAILED]\tNot all labels *.npz files were found")
        else:
            [print(f" - [FAILED]\tNot found {fn}") for fn in list_labels if not os.path.exists(fn)]


def get_argparse():
    desc = "This script checks the passed scene_list ensuring that all mvl-data defined at data_dir can be accessed."

    parser = argparse.ArgumentParser(
        description=desc,
        epilog=EPILOG
    )

    parser.add_argument(
        '-d', '--data_dir',
        default=f"{DEFAULT_MVL_DIR}",
        type=str,
        help=f'Output directory by default it will store at {DEFAULT_MVL_DIR}.'
    )
    
    parser.add_argument(
        '-f', '--scene_list',
        required=True,
        type=str,
        help=f'Scene list as JSON file. See {SCENE_LIST_DIR}.'
    )
    
    parser.add_argument(
        '-v',
        action="store_true",
        help=f'Explicit list of failures.'
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_argparse()
    main(args)
    