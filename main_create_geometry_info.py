import argparse
from mvl_datasets import CFG_DIR, ASSETS_DIR
import os
from mvl_datasets.config.cfg import read_omega_cfg
from mvl_datasets.utils.io_utils import get_files_given_a_pattern
from mvl_datasets.datasets.rgbd_datasets import RGBD_Dataset
from mvl_datasets.pre_processing.geometry_info import get_geometry_info
from mvl_datasets.utils.io_utils import save_json_dict, create_directory
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
    list_scenes_dir = get_files_given_a_pattern(
    args.scene_dir, flag_file="frm_ref.txt", exclude=["rgb", 'depth'])
    create_directory(cfg.output_dir, delete_prev=False)

    for scene_dir in list_scenes_dir:
        cfg.dataset.scene_dir = scene_dir
        compute_and_save_geometry_info(cfg)
    
def get_cfg(args):
    cfg = read_omega_cfg(args.cfg)
    cfg.dataset = dict()
    cfg.dataset.scene_dir = args.scene_dir
    cfg.output_dir = args.output_dir
    return cfg
        
def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--scene_dir',
        # required=True,
        default="/media/public_dataset/MP3D_360_FPE/MULTI_ROOM_SCENES/",
        type=str,
        help='scene directory'
    )
    
    parser.add_argument(
        '--output_dir',
        # required=True,
        default=f"{ASSETS_DIR}/geometry_info/",
        type=str,
        help='scene directory'
    )
    
    parser.add_argument(
        '--cfg',
        default=f"{CFG_DIR}/camera_height.yaml",
        help='Cfg to compute camera height'
    )
    
    args = parser.parse_args()
    return args
    

if __name__ == '__main__':
    args = get_args()
    main(args)
    
    