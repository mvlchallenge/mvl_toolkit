"""
This script allows us to zip the MP3D_FPE and HM3D-MVL dataset into several zip files separated by 
categories and scenes for convenience. i.e., 
- $SCENE_$VERSION_geometry.zip: Files which define geometries: cam poses, labels (floor plan only), selected key-frame
- $SCENE_$VERSION_rgb.zip: Images defined per scene
- $SCENE_$VERSION_depth.zip: Depth maps defined per scene
- $SCENE_$VERSION_npy.zip: Pre-computed estimation using HorizonNet (some scene also content from HohoNet)
Usage:
    python zip.py -s ${where the scene are stored} -o {where the zip files will be stored} -k [options: ["geom", "rgb", "depth", "npy"],] 
Example
    python mp3d_fpe/zip.py -s /MP3D_360_FPE/SINGLE_ROOM_SCENES/ -o MP3D_360_FPE/zipped_mp3d_fpe/
"""

import argparse
import os
import zipfile
from fileinput import filename

from tqdm import tqdm

from mvl_challenge.utils.io_utils import (create_directory,
                                         get_files_given_a_pattern,
                                         read_csv_file)


def zip_geometry_files(list_scenes, args):
    geometry_files = [
        'label.json', 'frm_ref.txt', 'keyframe_list.txt', "cam_pose_gt.csv",
        "cam_pose_estimated.csv"
    ]
    for scene in list_scenes:
        zip_filename = os.path.join(args.o,
                                    f"{scene.replace('/', '_')}_geometry.zip")
        scene_dir = os.path.join(args.scene_dir, scene)
        with zipfile.ZipFile(file=zip_filename, mode='w') as zf:
            list_fn = []
            for fn in geometry_files:
                list_fn += get_files_given_a_pattern(
                    data_dir=os.path.join(args.scene_dircene_dir, scene),
                    flag_file=fn,
                    exclude=["depth", "hn_mp3d"],
                    include_flag_file=True)
            list_arc_fn = process_arcname(list_fn, scene_dir)
            [
                zf.write(os.path.join(scene_dir, fn),
                         compress_type=zipfile.ZIP_STORED,
                         arcname=os.path.join(scene, fn))
                for fn in tqdm(list_arc_fn,
                               desc=f"zipping {os.path.join(scene, fn)}")
            ]


def get_list_frames(scene_path):
    key_fr_fn = get_files_given_a_pattern(data_dir=scene_path,
                                          flag_file='keyframe_list.txt',
                                          exclude=["depth", "hn_mp3d"],
                                          include_flag_file=True)[0]

    return read_csv_file(key_fr_fn)


def zip_rgb_files(list_scenes, args):
    for scene in list_scenes:
        zip_filename = os.path.join(args.o,
                                    f"{scene.replace('/', '_')}_rgb.zip")
        scene_dir = os.path.join(args.scene_dir, scene)
        with zipfile.ZipFile(file=zip_filename, mode='w') as zf:
            list_fn = get_list_frames(scene_dir)
            [
                zf.write(os.path.join(scene_dir, 'rgb', f"{fn}.png"),
                         compress_type=zipfile.ZIP_STORED,
                         arcname=os.path.join(scene, 'rgb', f"{fn}.png"))
                for fn in tqdm(list_fn,
                               desc=f"zipping {os.path.join(scene, 'rgb')}")
            ]


def zip_depth_files(list_scenes, args):
    for scene in list_scenes:
        zip_filename = os.path.join(args.o,
                                    f"{scene.replace('/', '_')}_depth.zip")
        scene_dir = os.path.join(args.scene_dir, scene)
        with zipfile.ZipFile(file=zip_filename, mode='w') as zf:
            list_fn = get_list_frames(scene_dir)
            [
                zf.write(os.path.join(scene_dir, 'depth', 'tiff',
                                      f"{fn}.tiff"),
                         compress_type=zipfile.ZIP_STORED,
                         arcname=os.path.join(scene, 'depth', 'tiff',
                                              f"{fn}.tiff"))
                for fn in tqdm(list_fn,
                               desc=f"zipping {os.path.join(scene, 'depth')}")
            ]


def zip_npy_files(list_scenes, args):
    for scene in list_scenes:
        zip_filename = os.path.join(args.o,
                                    f"{scene.replace('/', '_')}_npy.zip")
        scene_dir = os.path.join(args.scene_dir, scene)
        with zipfile.ZipFile(file=zip_filename, mode='w') as zf:
            list_fn = get_files_given_a_pattern(data_dir=scene_dir,
                                                flag_file=".npy",
                                                exclude=["depth", "rgb"],
                                                include_flag_file=True)

            [
                zf.write(fn,
                         compress_type=zipfile.ZIP_STORED,
                         arcname=os.path.relpath(fn, start=scene_dir))
                for fn in tqdm(list_fn, desc=f"zipping npy files {scene}")
            ]


def process_arcname(list_fn, base_dir):
    return [os.path.relpath(fn, start=base_dir) for fn in list_fn]


def zip_mvl_data(args):

    # ! Create output directory
    create_directory(args.o, delete_prev=False)

    print(f"Identifying mvl scenes in {args.scene_dir}.")
    list_scenes = get_files_given_a_pattern(
        args.scene_dir, "minos_poses.txt", exclude=["depth", 'rgb', "hn_mp3d"])
    list_arcname = process_arcname(list_scenes, base_dir=args.scene_dir)
    print(f"Found {list_arcname.__len__()} mvl scenes.")

    if "npy" in args.keys:
        zip_npy_files(list_arcname, args)

    if "rgb" in args.keys:
        zip_rgb_files(list_arcname, args)

    if "geom" in args.keys:
        zip_geometry_files(list_arcname, args)

    if "depth" in args.keys:
        zip_depth_files(list_arcname, args)


def get_argparse():
    parser = argparse.ArgumentParser()

    # * Input Directory (-s)
    parser.add_argument(
        '--scene_dir',
        # required=True,
        default="/media/public_dataset/MP3D_360_FPE/MULTI_ROOM_SCENES",
        type=str,
        help='Input Directory (-source)')

    # * Output Directory (-o)
    parser.add_argument(
        '-output',
        # required=True,
        default="/media/public_dataset/MP3D_360_FPE/zipped_mp3d_fpe",
        type=str,
        help='Output Directory (-o)')

    parser.add_argument(
        '-keys',
        # required=True,
        default=["geom", "rgb", "depth", "npy"],
        # default="rgb",
        nargs='+',
        help='Key argument in source ["geom", "rgb", "depth", "npy"]')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_argparse()
    zip_mvl_data(args)
