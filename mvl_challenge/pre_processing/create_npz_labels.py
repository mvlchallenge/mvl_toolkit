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
from mvl_challenge.utils.io_utils import get_scene_room_from_scene_room_idx, get_all_frames_from_scene_list
from mvl_challenge.utils.layout_utils import get_boundary_from_list_corners
from mvl_challenge.utils.spherical_utils import xyz2uv, uv2phi_coords, phi_coords2uv, phi_coords2xyz
from mvl_challenge.utils.vispy_utils import plot_color_plc
from mvl_challenge.utils.geometry_utils import tum_pose2matrix44, extend_array_to_homogeneous
from mvl_challenge.utils.image_utils import draw_boundaries_uv, draw_boundaries_phi_coords
from mvl_challenge.utils.image_utils import add_caption_to_image
from mvl_challenge.pre_processing.create_scene_room_idx_list import save_scene_list_from_mvl_directory


def save_phi_bound(args, list_corners, scene_room_idx_list):
    # ! ceiling
    crn_ceil = [np.array(c[0]).T for c in list_corners]
    bound_ceil  = get_boundary_from_list_corners(crn_ceil)
    
    # ! Floor
    crn_floor = [np.array(c[1]).T for c in list_corners]
    bound_floor  = get_boundary_from_list_corners(crn_floor)
    
    gt_dir = create_directory(args.output_dir, delete_prev=False)
    gt_vis_dir = create_directory(args.output_dir + "_vis", delete_prev=False)
     
    for sc_rm_idx in scene_room_idx_list:
        scene_room_idx = Path(sc_rm_idx).stem
        geom_info_fn = os.path.join(args.geom_info_dir, f"{sc_rm_idx}.json")
        assert os.path.exists(geom_info_fn), f"Not found {geom_info_fn}"
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
        if phi_coords_ceil is None or phi_coords_floor is None:
            logging.warning("phi_coords is None")
            continue
        phi_coords = np.vstack([phi_coords_ceil, phi_coords_floor])
        assert phi_coords.shape == (2, 1024)
        img = add_caption_to_image(
                image=img,
                caption="mvl-challenge " + scene_room_idx 
            )
    
        draw_boundaries_phi_coords(img, phi_coords)
          
        gt_vis_fn = os.path.join(gt_vis_dir, f"{scene_room_idx}.jpg")
        gt_fn = os.path.join(gt_dir, f"{scene_room_idx}")    
           
        imwrite(gt_vis_fn, img)
        np.savez_compressed(gt_fn, phi_coords=phi_coords)
        print(f"Saving... {gt_fn}.npz", end="\r")
        # logging.info(f"Saved {gt_fn}")
    print(f"Finished {gt_dir}")
    
def main(args):
    # ! Reading geometry info
    set_loggings()
    
    create_directory(args.output_dir, delete_prev=False)
    scene_room_idx_list = get_all_frames_from_scene_list(args.scene_list)    
    
    all_rooms = np.unique([get_scene_room_from_scene_room_idx(room) for room in scene_room_idx_list]).tolist()
    scene_rooms =  np.unique(["_".join(f.split("_")[:2]) for f in all_rooms]).tolist()
    for scene in tqdm(scene_rooms, desc="Loading scenes..."):
        list_room_per_scene = [r for r in all_rooms if scene in r]
        list_room_per_scene = sorted(list_room_per_scene, key=lambda x:int(x.split("room")[-1]))
        for room_name in list_room_per_scene:
            #! idx must coincide with {scene}_{ver}_room{idx}_{idx_fr}
            idx = int(room_name.split("room")[-1])
            scene_name, version, room = Path(room_name).stem.split("_")
            scene_dir = os.path.join(args.scene_dir, scene_name, version)
            
            mvl_labels_fn = os.path.join(scene_dir, 'mvl_challenge_labels.json')
            if os.path.exists(mvl_labels_fn):
                logging.info(f"Saving GT labels for {room_name}")
                #! We process only scenes with mvl-labels

                mvl_data = json.load(open(mvl_labels_fn, "r"))
                list_corners = mvl_data['room_corners'][idx]
                
                save_phi_bound(args, list_corners, [g for g in scene_room_idx_list if room_name in g])
        
    # ! Save scene_list for GT labels
    cfg = get_empty_cfg()
    cfg.mvl_dir = args.output_dir
    cfg.output_dir = Path(args.output_dir).parent.__str__()
    cfg.output_filename = "gt_labels__scene_list"
    save_scene_list_from_mvl_directory(cfg)
    
            
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
        default="/media/public_dataset/MP3D_360_FPE/MULTI_ROOM_SCENES/",
        # default="/media/public_dataset/HM3D-MVL/test",
        type=str,
        help='RGBD dataset directory.'
    )

    parser.add_argument(
        '-g', '--geom_info_dir',
        # required=True,
        default=f"{ASSETS_DIR}/issue_omitted_frames/geometry_info",
        type=str,
        help='Geometry information directory.'
    )

    parser.add_argument(
        '-f', '--scene_list',
        # required=True,
        default=f"{ASSETS_DIR}/issue_omitted_frames/test_scene_list.json",
        type=str,
        help='Scene List.'
    )
    
    parser.add_argument(
        '-o', '--output_dir',
        # required=True,
        default=f"{ASSETS_DIR}/issue_omitted_frames/labels/gt",
        type=str,
        help='Output directory for the output_file to be created.'
    )
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_argparse()
    main(args)
    