import argparse
from mvl_challenge.config.cfg import read_omega_cfg
from mvl_challenge import ASSETS_DIR, CFG_DIR, EPILOG
from mvl_challenge.datasets.rgbd_datasets import MP3D_FPE, RGBD_Dataset
from mvl_challenge.pre_processing.camera_height_utils import estimate_camera_height
from mvl_challenge.pre_processing.camera_height_per_rooms import estimate_cam_height_per_room
from mvl_challenge.utils.io_utils import get_idx_from_scene_room_idx
from mvl_challenge.utils.geometry_utils import get_quaternion_from_matrix
import logging
import os
import json


def get_geometry_info(dt: RGBD_Dataset):
    # ! Estimate camera height per room
    cam_h_dict = estimate_cam_height_per_room(dt.cfg.cam_height_cfg, dt)

    # ! List all frames in the scene
    scenes_room_fn = dt.cfg.dataset.scene_list
    scenes_room = json.load(open(scenes_room_fn, "r"))

    # ! list of Frames defined (defined in wc)
    list_fr = dt.get_list_frames()

    # ! Loop for every room identified
    list_geom_info = []
    for room_name in cam_h_dict.keys():

        # ! Cam height in WC
        cam_h_wc = cam_h_dict[room_name]['cam_h_wc']

        # ! List fr defined in this room
        list_fr_names = scenes_room[room_name]
        list_idx = sorted([get_idx_from_scene_room_idx(fr_names) for fr_names in list_fr_names])
        list_fr_room = sorted([fr for fr in list_fr if fr.idx in list_idx], key=lambda x: x.idx)

        # ! List geometry info per fr
        list_geom_info.append([{f"{room_name}_{fr.idx}":{
            'translation': [t for t in fr.pose[:3, 3]],
            'quaternion': [q for q in get_quaternion_from_matrix(fr.pose)],
            'cam_h':  cam_h_wc - fr.pose[1, 3]
        }}
            for fr in list_fr_room
        ])

    return list_geom_info
        

def main(args):
    cfg = read_omega_cfg(args.cfg)
    cfg.dataset = dict()
    cfg.dataset.scene_dir = args.scene_dir
    dt = RGBD_Dataset.from_cfg(cfg)
    geo_info_dict = get_geometry_info(dt)


def get_argparse():
    desc = "This script computes the geometry information per frame from a given MV dataset. " + \
        "The geometry info is defined as the geometrical information which define each frame , i.e., camera pose and camera height."

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
        help='MVL data scene directory.'
    )

    parser.add_argument(
        '-f', '--scene_list',
        # required=True,
        default=f"{ASSETS_DIR}/mvl_data/mp3d_fpe_scenes.json",
        type=str,
        help='Scene list file which contents all frames encoded in scene_room_idx format.'
    )

    parser.add_argument(
        '-o', '--output_dir',
        # required=True,
        default=f"{ASSETS_DIR}/mvl_data/geometry_info",
        type=str,
        help='Output directory for the output_file to be created.'
    )
    
    parser.add_argument(
        '--cfg',
        default=f"{CFG_DIR}/hyperparameters_camera_height.yaml",
        help=f'Hypermeter cfg (default: {CFG_DIR}/hyperparameters_camera_height.yaml )'
    )

    args = parser.parse_args()
    return args
    
if __name__ == '__main__':
    args = get_argparse()
    main(args)
