import argparse
import os
import gdown
import pandas as pd
from mvl_challenge import ASSETS_DIR, ROOT_DIR
from mvl_challenge.utils.io_utils import create_directory, read_txt_file
from mvl_challenge.config.cfg import set_loggings
from mvl_challenge import EPILOG
from mvl_challenge.utils.download_utils import download_file_from_google_drive
from tqdm import tqdm
import threading
from google_drive_downloader import GoogleDriveDownloader as gdd
import subprocess

def download_dirs(args):
    set_loggings()
    create_directory(args.output_dir, delete_prev=False)

    list_google_scenes = args.ids_file
    lines = read_txt_file(list_google_scenes)

    for l in lines:
        gd_id, dir_path = [l for l in l.replace(" ", ",").split(",") if l != ""][:2]
        output_dir = os.path.join(args.output_dir)
        count = f"{lines.index(l)+1}/{lines.__len__()}"
        print(f"Downloading...{count} {output_dir}")
        gdown.download_folder(id=gd_id, output=output_dir, quiet=False)

def callback_curl_downloading(gd_id, output_dir):
    subprocess.run(
        [
            "sh",
            f"{ROOT_DIR}/remote_data/download_gdrive_ids_file.sh",
            f"{gd_id}",
            f"{output_dir}"]
    )

def download_file_by_threads(args):
    set_loggings()
    create_directory(args.output_dir, delete_prev=False)

    list_threads = []
    list_google_scenes = args.ids_file
    lines = read_txt_file(list_google_scenes)

    for l in lines:
        gd_id, zip_fn = [l for l in l.replace(" ", ",").split(",") if l != ""][:2]
        output_file = os.path.join(args.output_dir, zip_fn)
        list_threads.append(
            threading.Thread(
                # target=download_google_drive_link,
                # args=(gd_id, output_file, f"{lines.index(l)+1}/{lines.__len__()}"))
            # threading.Thread(
            #     target=download_file_from_google_drive,
            #     args=(gd_id, output_file))
            target=callback_curl_downloading, 
            args=(gd_id, output_file))
            )
    [t.start() for t in list_threads]
    [t.join() for t in list_threads]


def download_file(args):
    set_loggings()
    create_directory(args.output_dir, delete_prev=False)

    list_google_scenes = args.ids_file
    lines = read_txt_file(list_google_scenes)

    for l in lines:
        gd_id, zip_fn = [l for l in l.replace(" ", ",").split(",") if l != ""][:2]
        output_file = os.path.join(args.output_dir, zip_fn)
        download_google_drive_link(
            gd_id, output_file, f"{lines.index(l)+1}/{lines.__len__()}"
        )



def download_google_drive_link_by_request(gd_id, output_file, count=""):
    print(f"Downloading...{count} {output_file}")
    gdd.download_file_from_google_drive(file_id=gd_id,
                                    dest_path=output_file,
                                    unzip=False)


def download_google_drive_link(gd_id, output_file, count=""):
    print(f"Downloading...{count} {output_file}")
    url = f"https://drive.google.com/uc?id={gd_id}"
    gdown.download(url, output_file, quiet=False)
    # url= f"https://docs.google.com/uc?export=download&confirm=t&id={gd_id}"
    # wget.download(url, out=output_file)
    # download_file_from_google_drive(gd_id, output_file)


def get_argparse():
    desc = (
        "This script Download a set of zip files corresponding to the mvl-data. "
        + "This zip files may content geometry_info files, images files, or/and gt npz labels files."
    )

    parser = argparse.ArgumentParser(description=desc, epilog=EPILOG)

    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        # required=True,
        default=f"{ASSETS_DIR}/tmp/downloaded_data",
        help="Output dataset directory.",
    )

    parser.add_argument(
        "-f",
        "--ids_file",
        type=str,
        # required=True,
        default="/media/NFS/kike/360_Challenge/mvl_toolkit/mvl_challenge/data/gdrive_files/gdrive_ids__warm_up_training_set_folders.csv",
        help="lists of IDS to download from GoogleDrive",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_argparse()
    # download_scenes(args)
    download_dirs(args)
