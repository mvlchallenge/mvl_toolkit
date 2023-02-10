import argparse
from mvl_datasets.datasets.rgbd_datasets import MP3D_FPE
from mvl_datasets.config.cfg import get_empty_cfg
from mvl_datasets.pre_processing.camera_height_from_pcl import estimate_camera_height
import numpy as np 
from mvl_datasets.utils.vispy_utils import plot_color_plc
from mvl_datasets.utils.io_utils import save_json_dict, create_directory, get_files_given_a_pattern
from tqdm import tqdm
import os
from copy import copy


def compute_cam_height_for_multiple_scene(args):
    list_scenes = get_files_given_a_pattern(data_dir=args.dataset_dir, flag_file="frm_ref.txt")
    create_directory(args.output_dir, delete_prev=False)
    
    for scene_dir in tqdm(list_scenes, desc="Reading scenes... "):
        cfg = get_empty_cfg()
        cfg.dataset = dict()
        cfg.dataset.scene_dir = scene_dir
        dt = MP3D_FPE.from_cfg(cfg) 
        compute_and_save_cam_height(args, dt)
        
def compute_and_save_cam_height(args, dt):
    cam_height_dict = dict()
    # org_fr = dt.get_list_frames()
    for list_fr in dt.iter_rooms_scenes():
        # Each fr in list_fr is wrt to room references
        room_name=list_fr[0].room_name
        init_idx = list_fr[0].idx
        cam_h = estimate_camera_height(args, list_fr)
        
        # h_room2world = [fr.pose[1, 3] for fr in org_fr if fr.idx == init_idx][0]
        # cam_h2world =  cam_h + h_room2world 
        
        # cam_height_dict[room_name]=dict(
        #     cam_height=cam_h2world, 
        #     first_frm_idx=init_idx
        #     )

        cam_height_dict[room_name]=dict(
            cam_height=cam_h, 
            first_frm_idx=init_idx
            )
        
        create_directory(args.output_dir, delete_prev=False)
        camera_height_fn = os.path.join(args.output_dir, f"{dt.scene_name}_cam_height.json")
        save_json_dict(camera_height_fn, cam_height_dict)
 
def compute_cam_height_for_one_scene(args):
    cfg = get_empty_cfg()
    cfg.dataset = dict()
    cfg.dataset.scene_dir = args.scene_dir
    dt = MP3D_FPE.from_cfg(cfg)
    compute_and_save_cam_height(args, dt)
        
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
        '--output_dir',
        # required=True,
        default="./assets",
        type=str,
        help='Output directory'
    )
    
    parser.add_argument(
        '--fit_error',
        # required=True,
        default=0.001,
        help='How much variance is allowed for RANSAC plane estimation'
    )
    
    parser.add_argument(
        '--xz_distance',
        # required=True,
        default=1,
        help='Max distance in XZ around the 1st camera frame'
    )
    
    parser.add_argument(
        '--min_height',
        # required=True,
        default=1,
        help='Min camera height from camera frames to floor'
    )  
    
    parser.add_argument(
        '--min_samples',
        # required=True,
        default=200,
        help='Min number of point used for RANSAC plane estimation'
    )

    parser.add_argument(
        '--r',
        action='store_true',
        help='Min number of point used for RANSAC plane estimation'
    )

    parser.add_argument(
        '--iter',
        default=5,
        help='Number of random iteration to compute camera height'
    )
    
    args = parser.parse_args()
    return args
    

if __name__ == '__main__':
    args = get_args()
    if args.r:
        compute_cam_height_for_multiple_scene(args)
    else:
        compute_cam_height_for_one_scene(args)
    