import argparse
from mvl_datasets.config.cfg import read_omega_cfg
from mvl_datasets import ASSETS_DIR, CFG_DIR, MP3D_FPE_DATA_DIR
from mvl_datasets.datasets.rgbd_datasets import MP3D_FPE, RGBD_Dataset
from mvl_datasets.pre_processing.camera_height import estimate_camera_height
from mvl_datasets.pre_processing.camera_height_per_rooms import estimate_cam_height_per_room
from mvl_datasets.utils.io_utils import get_idx_from_fr_name
from mvl_datasets.utils.geometry_utils import get_quaternion_from_matrix
import logging
import os
import json


def get_geometry_info_for_mp3d_fpe(dt: MP3D_FPE):
    # ! Estimate camera height per room
    cam_h_dict = estimate_cam_height_per_room(dt.cfg.cam_height_cfg, dt)

    # ! List all frames in the scene
    scenes_room_fn = os.path.join(MP3D_FPE_DATA_DIR, "mp3d_fpe_room_scenes.json")
    scenes_room = json.load(open(scenes_room_fn, "r"))

    # ! list of Frames defined (defined in wc)
    list_fr = dt.get_list_frames()

    # ! Loop for every room identified
    for room_name in cam_h_dict.keys():

        # ! Cam height in WC
        cam_h_wc = cam_h_dict[room_name]['cam_h_wc']

        # ! List fr defined in this room
        list_fr_names = scenes_room[room_name]
        list_idx = sorted([get_idx_from_fr_name(fr_names) for fr_names in list_fr_names])
        list_fr_room = sorted([fr for fr in list_fr if fr.idx in list_idx], key=lambda x: x.idx)

        # ! List geometry info per fr
        list_geom_info = [{f"{room_name}_{fr.idx}":{
            'translation': [t for t in fr.pose[:3, 3]],
            'quaternion': [q for q in get_quaternion_from_matrix(fr.pose)],
            'cam_h':  cam_h_wc - fr.pose[1, 3]
        }}
            for fr in list_fr_room
        ]

        return list_geom_info
        

def get_geometry_info(dt: RGBD_Dataset):
    if dt.__str__() == "MP3D_FPE":
        geo_info = get_geometry_info_for_mp3d_fpe(dt)
    elif dt.__str__() == "HM3D-MVL":
        # ! Data from HM3D-MVL
        pass
        # here the dict must be
        """
        list([scene_name]_[version/room]_[fr]: {
            translation: value, 
            quaternion: value,
            cam_h: value
            })
        """
    else:
        logging.error(f"dt class unknown {dt.__str__()}")
        raise NotImplementedError()

    return geo_info


def main(args):
    cfg = read_omega_cfg(args.cfg)
    cfg.dataset = dict()
    cfg.dataset.scene_dir = args.scene_dir
    dt = RGBD_Dataset.from_cfg(cfg)
    geo_info_dict = get_geometry_info(dt)


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
