import argparse
from mvl_challenge import CFG_DIR, ASSETS_DIR, EPILOG
import os
from mvl_challenge.config.cfg import read_omega_cfg
from mvl_challenge.utils.io_utils import get_files_given_a_pattern
from mvl_challenge.datasets.rgbd_datasets import RGBD_Dataset
from mvl_challenge.pre_processing.geometry_info_utils import get_geometry_info
from mvl_challenge.utils.io_utils import save_json_dict, create_directory
from tqdm import tqdm
import logging


def save_geometry_info(cfg, data_dict):
    for dt in data_dict:
        scene_name = list(dt.keys())[0]
        geom_info  = dt[scene_name] 
        fn = os.path.join(cfg.output_dir, f"{scene_name}.json")
        save_json_dict(fn, geom_info)


def compute_and_save_geometry_info(cfg):
    logging.info(f"Saving geometric info for {cfg.dataset.scene_dir}")
    dt = RGBD_Dataset.from_cfg(cfg)
    geom_info = get_geometry_info(dt)
    #! save geometry info
    [save_geometry_info(cfg, data) for data in geom_info]


def main(args):
    cfg = get_cfg(args)
    # ! Getting all scenes path define the passed scene_dir
    list_scenes_dir = get_files_given_a_pattern(
    args.scene_dir, flag_file="frm_ref.txt", exclude=["rgb", 'depth'])
    
    assert list_scenes_dir.__len__() > 0, "No scenes were found at {args.scene_dir} "
    
    create_directory(cfg.output_dir, delete_prev=False)

    for scene_dir in tqdm(list_scenes_dir, desc="list scenes..."):
        cfg.dataset.scene_dir = scene_dir
        cfg.dataset.scene_list = args.scene_list
        compute_and_save_geometry_info(cfg)
    
    
def get_cfg(args):
    cfg = read_omega_cfg(args.cfg)
    cfg.dataset = dict()
    cfg.dataset.scene_dir = args.scene_dir
    cfg.dataset.scene_list = args.scene_list
    cfg.output_dir = args.output_dir
    return cfg
        
        
def get_argparse():
    desc = "This script computes the geometry information per frame from a given MV dataset. " + \
        "The geometry info is the geometrical information for each frame, i.e., camera pose and camera height."

    parser = argparse.ArgumentParser(
        description=desc,
        epilog=EPILOG
    )

    parser.add_argument(
        '-d', '--scene_dir',
        # required=True,
        # default="/media/public_dataset/MP3D_360_FPE/SINGLE_ROOM_SCENES/",
        # default="/media/public_dataset/HM3D-MVL/test/",
        default=None,
        type=str,
        help='RGBD dataset directory.'
    )

    parser.add_argument(
        '-f', '--scene_list',
        required=True,
        # default=f"{ASSETS_DIR}/mvl_data/hm3d_mvl_test_scene_list.json",
        type=str,
        help='Scene list file which contents all frames encoded in scene_room_idx format.'
    )

    parser.add_argument(
        '-o', '--output_dir',
        # required=True,
        default=f"{ASSETS_DIR}/tmp/geometry_info",
        type=str,
        help='Output directory for the output_file to be created.'
    )
    
    parser.add_argument(
        '--cfg',
        default=f"{CFG_DIR}/camera_height.yaml",
        help=f'Hypermeter cfg (default: {CFG_DIR}/camera_height.yaml)'
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_argparse()
    main(args)
    
    