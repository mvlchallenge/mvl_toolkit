import argparse
import os
from mvl_challenge import EPILOG, ASSETS_DIR, DEFAULT_MVL_DIR, SCENE_LIST_DIR
from mvl_challenge.utils.check_utils import check_scene_list

def main(args):
    check_scene_list(args)


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

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_argparse()
    main(args)
    