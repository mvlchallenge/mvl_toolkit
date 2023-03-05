import argparse
from mvl_challenge import CFG_DIR, ASSETS_DIR, EPILOG
import os
from mvl_challenge.config.cfg import read_omega_cfg
from mvl_challenge.utils.io_utils import get_files_given_a_pattern
from mvl_challenge.datasets.rgbd_datasets import RGBD_Dataset
from mvl_challenge.pre_processing.geometry_info_utils import get_geometry_info
from mvl_challenge.utils.io_utils import save_json_dict, create_directory
from mvl_challenge.config.cfg import set_loggings
from tqdm import tqdm
import logging
from pathlib import Path
from shutil import copyfile
from  imageio import imread, imwrite

def main(args):
    # ! Reading geometry info
    set_loggings()
    list_geom_info = os.listdir(args.geom_info_dir)
    create_directory(args.output_dir, delete_prev=False)
    for geom_info in tqdm(list_geom_info, desc="list geom info..."):
        scene, version, _, idx = Path(geom_info).stem.split("_")
        src = os.path.join(args.scene_dir, scene, version, 'rgb', f'{idx}.jpg')
        dst = os.path.join(args.output_dir, f'{Path(geom_info).stem}.jpg')
        if os.path.exists(src):
            copyfile(src, dst)
            logging.info(f"Img {dst} processed...")
        else:
            src = os.path.join(args.scene_dir, scene, version, 'rgb', f'{idx}.png')
            if os.path.exists(src):
                img = imread(src)
                imwrite(dst, img)
                logging.info(f"Img {dst} processed...")
                continue                
        logging.info(f"Img {src} skipped...")
     
def get_argparse():
    desc = "This script creates a hard copy of the RGB images defined in the given MLV dataset. " + \
        "The RGB images are encoded in scene_room_idx format."

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
        '-g', '--geom_info_dir',
        # required=True,
        default=f"{ASSETS_DIR}/mvl_data/geometry_info",
        type=str,
        help='Geometry information directory. (default: {ASSETS_DIR}/mvl_data/geometry_info)'
    )
    
    parser.add_argument(
        '-o', '--output_dir',
        # required=True,
        default=f"{ASSETS_DIR}/mvl_data/img",
        type=str,
        help='Output directory for the output_file to be created.'
    )
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_argparse()
    main(args)
    
    