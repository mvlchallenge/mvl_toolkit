import os
import json
import argparse
from tqdm import tqdm
from pathlib import Path
import numpy as np
import logging
from imageio import imread, imwrite
from mvl_challenge import EPILOG, ASSETS_DIR, CFG_DIR
from mvl_challenge.utils.io_utils import create_directory
from mvl_challenge.config.cfg import set_loggings, get_empty_cfg
from mvl_challenge.utils.io_utils import get_scene_room_from_scene_room_idx
from mvl_challenge.utils.layout_utils import get_boundary_from_list_corners
from mvl_challenge.utils.spherical_utils import xyz2uv, uv2phi_coords, phi_coords2uv, phi_coords2xyz
from mvl_challenge.utils.vispy_utils import plot_color_plc
from mvl_challenge.utils.geometry_utils import tum_pose2matrix44, extend_array_to_homogeneous
from mvl_challenge.utils.image_utils import draw_boundaries_uv, draw_boundaries_phi_coords
from mvl_challenge.utils.image_utils import add_caption_to_image
from mvl_challenge.pre_processing.create_scene_room_idx_list import scene_list_from_mvl_directory


def save_phi_bound(args, list_corners, list_geom_info):
    # ! ceiling
    crn_ceil = [np.array(c[0]).T for c in list_corners[0]]
    bound_ceil  = get_boundary_from_list_corners(crn_ceil)
    
    # ! Floor
    crn_floor = [np.array(c[1]).T for c in list_corners[0]]
    bound_floor  = get_boundary_from_list_corners(crn_floor)
    
    gt_dir = create_directory(args.output_dir, delete_prev=False)
    gt_vis_dir = create_directory(args.output_dir + "_vis", delete_prev=False)
    
    for geom_info in tqdm(list_geom_info, desc="Save Gt labels..."):
        scene_room_idx = Path(geom_info).stem
        geom_info_fn = os.path.join(args.geom_info_dir, geom_info)
        geom_data = json.load(open(geom_info_fn, 'r'))
        
        # ! Getting SE3 transformation in geom_info
        cam_pose = tum_pose2matrix44([-1] +
            geom_data['translation'] + geom_data['quaternion'])
        
        # ! Transform into camera coordinates
        local_xyz_ceil = np.linalg.inv(cam_pose)[:3, :] @ extend_array_to_homogeneous(bound_ceil)
        local_xyz_floor = np.linalg.inv(cam_pose)[:3, :] @ extend_array_to_homogeneous(bound_floor)
        
        # ! projection into uv coord
        uv_ceil = xyz2uv(local_xyz_ceil)
        uv_floor  = xyz2uv(local_xyz_floor)
        
        img = imread(Path(args.geom_info_dir).parent.__str__() + f"/img/{scene_room_idx}.jpg")
        
        #! Projection into phi_coords
        phi_coords_ceil = uv2phi_coords(uv_ceil, type_bound='ceiling')
        phi_coords_floor = uv2phi_coords(uv_floor, type_bound='floor')
        phi_coords = np.vstack([phi_coords_ceil, phi_coords_floor])
        
        img = add_caption_to_image(
                image=img,
                caption="mvl-challenge " + scene_room_idx 
            )
    
        draw_boundaries_phi_coords(img, phi_coords)
          
        gt_vis_fn = os.path.join(gt_vis_dir, f"{scene_room_idx}.jpg")
        gt_fn = os.path.join(gt_dir, f"{scene_room_idx}")    
           
        imwrite(gt_vis_fn, img)
        np.savez_compressed(gt_fn, phi_coords=phi_coords)
        
        
def main(args):
    # ! Reading geometry info
    set_loggings()
    
    create_directory(args.output_dir, delete_prev=False)
    list_geom_info = os.listdir(args.geom_info_dir)
    
    list_rooms = np.unique([get_scene_room_from_scene_room_idx(room) for room in list_geom_info]).tolist()
    
    for room_name in list_rooms:
        scene_name, version, room = Path(room_name).stem.split("_")
        scene_dir = os.path.join(args.scene_dir, scene_name, version)
        
        mvl_labels_fn = os.path.join(scene_dir, 'mvl_challenge_labels.json')
        if os.path.exists(mvl_labels_fn):
            logging.info(f"Saving GT labels for {room_name}")
            #! We process only scenes with mvl-labels

            mvl_data = json.load(open(mvl_labels_fn, "r"))
            list_corners = mvl_data['room_corners']
            
            save_phi_bound(args, list_corners, [g for g in list_geom_info if room_name in g])
    
    # ! Save scene_list for GT labels
    cfg = get_empty_cfg()
    cfg.mvl_dir = args.output_dir
    cfg.output_dir = Path(args.output_dir).parent.__str__()
    cfg.output_filename = "gt_labels__scene_list"
    scene_list_from_mvl_directory(cfg)
    
            
def get_argparse():
    desc = "This script creates the npy files from the mvl-annotation. " + \
        "This npy files content the boundary defined for ceiling and floor."

    parser = argparse.ArgumentParser(
        description=desc,
        epilog=EPILOG
    )

    parser.add_argument(
        '-d', '--scene_dir',
        # required=True,
        default="/media/public_dataset/MP3D_360_FPE/SINGLE_ROOM_SCENES/",
        # default="/media/public_dataset/HM3D-MVL/test/BHXhpBwSMLh",
        type=str,
        help='RGBD dataset directory.'
    )

    parser.add_argument(
        '-g', '--geom_info_dir',
        # required=True,
        default="/media/public_dataset/mvl_challenge/mp3d_fpe/geometry_info",
        type=str,
        help='Geometry information directory.'
    )

    parser.add_argument(
        '-o', '--output_dir',
        # required=True,
        default=f"{ASSETS_DIR}/mvl_data/mp3d_fpe/labels/gt",
        type=str,
        help='Output directory for the output_file to be created.'
    )
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_argparse()
    main(args)
    