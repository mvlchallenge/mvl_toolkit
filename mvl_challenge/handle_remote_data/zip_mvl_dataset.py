import argparse
from mvl_challenge import EPILOG, CFG_DIR, ASSETS_DIR
from mvl_challenge.utils.check_utils import check_mvl_dataset
from mvl_challenge.utils.io_utils import create_directory
import json
import zipfile
import os
from mvl_challenge.handle_remote_data.zip_rgbd_dataset import process_arcname
from tqdm import tqdm
    
def zip_mvl_data(args): 
    #! Get scene list
    data_scene, ret = check_mvl_dataset(args)
    assert ret, "Failed checking MVL dataset."

    create_directory(args.output_dir, delete_prev=False)
    for room, scene_list in data_scene.items():    
        zip_filename = os.path.join(args.output_dir, f"{room}.zip")
        with zipfile.ZipFile(file=zip_filename, mode='w') as zf:
            geo_info_fn = [os.path.join(args.scene_dir, 'geometry_info', f"{sc}.json") for sc in scene_list]
            img_fn = [os.path.join(args.scene_dir, 'img', f"{sc}.jpg") for sc in scene_list]
            
            zip_data(args, zf, geo_info_fn)
            zip_data(args, zf, img_fn)
            
            
def zip_data(args, zf, geo_info_fn):
    list_arc_fn = process_arcname(geo_info_fn, args.scene_dir)
    [(print(f"zipping {fn}"),
    zf.write(os.path.join(args.scene_dir, fn),
                compress_type=zipfile.ZIP_STORED,
                arcname=fn))
    for fn in tqdm(list_arc_fn)
    ]  
       
       
def get_argparse():
    desc = "This script zip and unzip MVL dataset locally. " + \
        "A mvl dataset is composed of geometry info, and images. Both encoded in scene_room_idx format."
        
    parser = argparse.ArgumentParser(
        description=desc,
        epilog=EPILOG
    )

    parser.add_argument(
        '-d', '--scene_dir',
        # required=True,
        # default="/media/public_dataset/MP3D_360_FPE/SINGLE_ROOM_SCENES/",
        # default="/media/public_dataset/mvl_challenge/mp3d_fpe",
        # default=f'{ASSETS_DIR}/tmp/zip_files',
        # default=None,
        type=str,
        help='MVL dataset directory.'
    )

    parser.add_argument(
        '-f', '--scene_list',
        # required=True,
        default="/media/public_dataset/mvl_challenge/mp3d_fpe__single_room_scene_list.json",
        type=str,
        help='Scene list file which contents all frames encoded in scene_room_idx format.'
    )

    parser.add_argument(
        '-o', '--output_dir',
        # required=True,
        default=f"{ASSETS_DIR}/tmp/zip_files",
        type=str,
        help='Output directory for the output_file to be created.'
    )
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_argparse()
    zip_mvl_data(args)
  