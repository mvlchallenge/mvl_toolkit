import argparse
from mvl_challenge.datasets.rgbd_datasets import MP3D_FPE
from mvl_challenge.config.cfg import get_empty_cfg
from mvl_challenge.pre_processing.utils.camera_height_utils import estimate_camera_height
import numpy as np 
from mvl_challenge.utils.vispy_utils import plot_color_plc
from mvl_challenge.utils.io_utils import save_json_dict, create_directory, get_files_given_a_pattern
from mvl_challenge import ASSETS_DIR, CFG_DIR
from mvl_challenge.config.cfg import read_omega_cfg
import logging

        
def estimate_cam_height_per_room(cfg, dt: MP3D_FPE):
    list_fr2world = dt.get_list_frames()
    cam_height_dict = {}
    for list_fr in dt.iter_rooms_scenes():
        if list_fr.__len__() < 5:
            continue
        #! Each fr in list_fr is wrt to room references
        room_name=list_fr[0].room_name
        init_idx = list_fr[0].idx
        cam_h_rc = estimate_camera_height(cfg, list_fr)
        
        room_wc = [fr.pose[1, 3] for fr in list_fr2world if fr.idx == init_idx][0]
        cam_h_wc =  cam_h_rc + room_wc 
        cam_height_dict[room_name]=dict(
            cam_h_rc = cam_h_rc,
            cam_h_wc= cam_h_wc
        )
    [logging.info(f"Room: {r}\tcam_h(r): {d['cam_h_rc']:2.3f}\tcam_h(w): {d['cam_h_wc']:2.3f}") for r, d in cam_height_dict.items()]
    return cam_height_dict
 
def main(args):
    cfg = read_omega_cfg(args.cfg)
    cfg.dataset = dict()
    cfg.dataset.scene_dir = args.scene_dir
    dt = MP3D_FPE.from_cfg(cfg)
    estimate_cam_height_per_room(cfg.cam_height_cfg, dt)
        
def get_args():
    parser = argparse.ArgumentParser()
    
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
    