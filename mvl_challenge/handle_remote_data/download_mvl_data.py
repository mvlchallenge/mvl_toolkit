import argparse
import os
import gdown
import pandas as pd
from mvl_challenge import ASSETS_DIR, ROOT_DIR
from mvl_challenge.utils.io_utils import create_directory
from mvl_challenge.config.cfg import set_loggings
from tqdm import tqdm

def download_scenes(args):
    set_loggings()
    create_directory(args.output_dir, delete_prev=False)

    list_google_scenes = args.ids_file
    scenes_ids = pd.read_csv(list_google_scenes)

    for gd_id, zip_fn in zip(scenes_ids.Id, scenes_ids.Name):
        print(f"Downloading... {zip_fn}")
        url = f"https://drive.google.com/uc?id={gd_id}"
        output_file = os.path.join(args.output_dir, zip_fn)
        gdown.download(url, output_file, quiet=False)


def get_argparse():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '-o', '--output_dir', 
        type=str, 
        default=f"{ASSETS_DIR}/tmp/downloaded_data", 
        help='Output dataset directory.'
        )
    
    parser.add_argument(
        "-f", '--ids_file', 
        type=str, 
        default=f"{ROOT_DIR}/data/mp3d_fpe/test_google_drive_ids.csv", 
        help="lists of IDS to download from GoogleDrive"
        )
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_argparse()
    download_scenes(args)