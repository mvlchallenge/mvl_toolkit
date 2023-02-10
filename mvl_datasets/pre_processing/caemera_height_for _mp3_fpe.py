import argparse
from mvl_datasets.datasets.rgbd_datasets import MP3D_FPE
from mvl_datasets.config.cfg import get_empty_cfg
from mvl_datasets.pre_processing.camera_height_from_pcl import estimate_camera_height
import numpy as np 
from mvl_datasets.utils.vispy_utils import plot_color_plc
from mvl_datasets.utils.io_utils import save_json_dict, create_directory, get_files_given_a_pattern
from mvl_datasets import ASSETS_DIR, CFG_DIR
from mvl_datasets.config.cfg import read_omega_cfg
import logging

        
def compute_cam_height_per_room(cfg, dt):
    list_fr2world = dt.get_list_frames()
    caption_output = []
    for list_fr in dt.iter_rooms_scenes():
        #! Each fr in list_fr is wrt to room references
        room_name=list_fr[0].room_name
        init_idx = list_fr[0].idx
        cam_h = estimate_camera_height(cfg, list_fr)
        
        room2world = [fr.pose[1, 3] for fr in list_fr2world if fr.idx == init_idx][0]
        cam_h2world =  cam_h + room2world 
        caption_output.append(f"Room_fr: {room_name}_{init_idx}\tCam-h (room): {cam_h:2.3f}\tCam-h (world){cam_h2world:2.3f}")
    [logging.info(r) for r in caption_output]
        
 
def main(args):
    cfg = get_empty_cfg()
    cfg.dataset = dict()
    cfg.dataset.scene_dir = args.scene_dir
    dt = MP3D_FPE.from_cfg(cfg)
    cfg_cam = read_omega_cfg(args.cfg)
    compute_cam_height_per_room(cfg_cam, dt)
        
def get_args():
    parser = argparse.ArgumentParser()
    # * Input Directory (-s)
    
    parser.add_argument(
        '--scene_dir',
        # required=True,
        default="/media/public_dataset/MP3D_360_FPE/MULTI_ROOM_SCENES/2t7WUuJeko7/1/",
        type=str,
        help='Directory of all scene in the dataset'
    )
    
    parser.add_argument(
        '--cfg',
        default=f"{CFG_DIR}/camera_height.yaml",
        help='Cfg tp compute camera height'
    )
    
    args = parser.parse_args()
    return args
    

if __name__ == '__main__':
    args = get_args()
    main(args)
    