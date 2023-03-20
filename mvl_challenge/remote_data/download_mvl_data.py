import argparse
import os
# import gdown
import pandas as pd
from mvl_challenge import ASSETS_DIR, ROOT_DIR
from mvl_challenge.utils.io_utils import create_directory, read_txt_file
from mvl_challenge.config.cfg import set_loggings
from mvl_challenge import EPILOG
from mvl_challenge.utils.download_utils import download_file_from_google_drive
from tqdm import tqdm

def download_scenes(args):
    set_loggings()
    create_directory(args.output_dir, delete_prev=False)

    list_google_scenes = args.ids_file
    lines = read_txt_file(list_google_scenes)

    for l in lines:
        gd_id, zip_fn = [l for l in l.replace(" ",",").split(",") if l != ''][:2]  
        output_file = os.path.join(args.output_dir, zip_fn)
        download_google_drive_link(gd_id, output_file)


def download_google_drive_link(gd_id, output_file):
    print(f"Downloading... {output_file}")
    # url = f"https://drive.google.com/uc?id={gd_id}"
    # gdown.download(url, output_file, quiet=False)
    # url= f"https://docs.google.com/uc?export=download&confirm=t&id={gd_id}"
    # wget.download(url, out=output_file)
    download_file_from_google_drive(gd_id, output_file)


def get_argparse():
    desc = "This script Download a set of zip files corresponding to the mvl-data. " + \
        "This zip files may content geometry_info files, images files, or/and gt npz labels files."

    parser = argparse.ArgumentParser(
        description=desc,
        epilog=EPILOG
    )

    parser.add_argument(
        '-o', '--output_dir', 
        type=str,
        required=True, 
        # default=f"{ASSETS_DIR}/tmp/downloaded_data", 
        help='Output dataset directory.'
        )
    
    parser.add_argument(
        "-f", '--ids_file', 
        type=str, 
        required=True,
        # default=f"{ASSETS_DIR}/mvl_data/pilot__mp3d_fpe/zips/ids_1ORpSP60h34TIOZlYstkuKhQBDaloPR5M.csv", 
        help="lists of IDS to download from GoogleDrive"
        )
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_argparse()
    download_scenes(args)