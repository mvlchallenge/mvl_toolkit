
import argparse
import os
import logging
from tqdm import tqdm

from mvl_datasets.config.cfg import get_empty_cfg
from mvl_datasets.pre_processing.camera_height_from_pcl import get_args as default_args, get_camera_height
from mvl_datasets.datasets.rgbd_datasets import RGBD_Dataset
from mvl_datasets.utils.io_utils import create_directory, get_files_given_a_pattern, save_json_dict

def main(args):

    list_scenes = get_files_given_a_pattern(data_dir=args.dataset_dir, flag_file="frm_ref.txt")
    create_directory(args.output_dir, delete_prev=False)
    
    data = dict()
    for scene_dir in tqdm(list_scenes, desc="Reading scenes... "):
        cfg = get_empty_cfg()
        cfg.dataset = dict()
        cfg.dataset.scene_dir = scene_dir
        dt = RGBD_Dataset.from_cfg(cfg)
        list_fr = dt.get_list_frames()
        h = get_camera_height(args, list_fr)
        data[dt.scene_name] = h
        print(f"Scene {scene_dir} - H: {h}")        
        save_json_dict(
            os.path.join(args.output_dir, "camera_heights.json"), 
            data
        )
        
   
def get_args():
    parser = argparse.ArgumentParser()
    # * Input Directory (-s)
    parser.add_argument(
        '--dataset_dir',
        # required=True,
        default="/media/public_dataset/HM3D-MVL/test",
        type=str,
        help='Directory of all scene in the dataset'
    )
    
    parser.add_argument(
        '--output_dir',
        # required=True,
        default="./assets",
        type=str,
        help='Output directory'
    )
    
    args = default_args(parser)
    args = parser.parse_args()
    return args
        
  
if __name__ == '__main__':
    args = get_args()
    main(args)
    