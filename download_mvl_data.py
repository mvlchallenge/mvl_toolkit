from dataclasses import dataclass
import os
import subprocess
import argparse
from pathlib import Path
from mvl_challenge import (
    ASSETS_DIR,
    ROOT_DIR,
    EPILOG,
    CFG_DIR,
    GDRIVE_DIR,
    DEFAULT_DOWNLOAD_DIR,
)
from mvl_challenge.config.cfg import get_empty_cfg, read_omega_cfg
from mvl_challenge.remote_data.download_mvl_data import download_file, download_dirs, download_file_by_threads
from mvl_challenge.utils.io_utils import create_directory, save_compressed_phi_coords
from mvl_challenge.datasets.mvl_dataset import MVLDataset, iter_mvl_room_scenes
from mvl_challenge.models.wrapper_horizon_net import WrapperHorizonNet
from mvl_challenge.challenge_results.create_zip_results import zip_results
from mvl_challenge.remote_data.download_mvl_data import download_google_drive_link


@dataclass
class DataSplit:
    GDRIVE_IDS_MVL_DATA_FN: str
    GDRIVE_IDS_LABELS_FN: str
    TYPE: str
    GDRIVE_ID: str
    GDRIVE_ID_LABELS: str
    GT_LABELS: bool


def download_data_split_by_folders(args, data_split: DataSplit):
    #! Downloading mvl data
    zip_dir = os.path.join(args.output_dir, "zips", data_split.TYPE)
    create_directory(zip_dir, delete_prev=False)

    # tmp_dir = os.path.join(zip_dir, "tmp_dir")
    # create_directory(tmp_dir, delete_prev=True, ignore_request=True)
    cfg = get_empty_cfg()
    cfg.output_dir = zip_dir
    cfg.ids_file = os.path.join(GDRIVE_DIR, data_split.GDRIVE_IDS_MVL_DATA_FN)
    download_dirs(cfg)

    # list_dir_path = os.listdir()
    # # ! Unzipping mvl-data
    output_dir = os.path.join(args.output_dir, "mvl_data")
    unzip(zip_dir, output_dir)


def download_data_split(args, data_split: DataSplit):
    #! Downloading mvl data
    zip_dir = os.path.join(args.output_dir, "zips", f"{data_split.TYPE}")
    create_directory(zip_dir, delete_prev=False)
    zip_filename = os.path.join(args.output_dir, "zips", f"{data_split.TYPE}.zip")
    # download_gdrive_file(data_split.GDRIVE_IDS_MVL_DATA_FN, zip_dir)
    download_entire_zip_split(data_split.GDRIVE_ID, zip_filename)

    # # ! Unzipping mvl-data
    output_dir = os.path.join(args.output_dir, "mvl_data")
    unzip(zip_dir, output_dir)

    if data_split.GT_LABELS:
        zip_dir = os.path.join(args.output_dir, "zips", "labels", f"{data_split.TYPE}")
        create_directory(zip_dir, delete_prev=False)
        # download_gdrive_file(data_split.GDRIVE_IDS_LABELS_FN, zip_dir)
        zip_filename = os.path.join(args.output_dir, "zips", "labels", f"{data_split.TYPE}.zip")
        download_entire_zip_split(data_split.GDRIVE_ID_LABELS, zip_filename)

        output_dir = os.path.join(args.output_dir, "mvl_data")
        unzip(zip_dir, output_dir)

    print(f"** \tzip dir for {data_split.TYPE}:\t\t{zip_dir}")
    print(f"** \tmvl dir for {data_split.TYPE}:\t\t{output_dir}")
    print(f"*\t->>>\t{data_split.TYPE} downloaded successfully\t<<<-\t*")


def download_entire_zip_split(gdrive_id, zip_filename):
    download_google_drive_link(gd_id=gdrive_id, output_file=zip_filename)
    subprocess.run(
        [
            "unzip",
            f"{zip_filename}",
            "-d",
            f"{Path(zip_filename).parent}"
        ]
    )


def unzip(zip_dir, output_dir):
    create_directory(output_dir, delete_prev=False)
    subprocess.run(
        [
            "bash",
            f"{ROOT_DIR}/remote_data/unzip_data.sh",
            "-d",
            f"{zip_dir}",
            "-o",
            f"{output_dir}",
        ]
    )


def download_gdrive_file(gdrive_fn, zip_dir):
    create_directory(zip_dir, delete_prev=False)
    cfg = get_empty_cfg()
    cfg.output_dir = zip_dir
    cfg.ids_file = os.path.join(GDRIVE_DIR, gdrive_fn)
    download_file_by_threads(cfg)


def main(args):
    if args.split == "pilot":
        data_split = DataSplit(
            GDRIVE_IDS_MVL_DATA_FN=os.path.join(
                GDRIVE_DIR, "gdrive_ids__pilot_set.csv"
            ),
            GDRIVE_IDS_LABELS_FN=os.path.join(
                GDRIVE_DIR, "gdrive_ids__pilot_labels.csv"
            ),
            TYPE="pilot_set",
            GT_LABELS=True,
            GDRIVE_ID="13dFArf0oKznUsOZTkumjspRT8Z2sTqHb",
            GDRIVE_ID_LABELS="1F5QW0QpoxublJTA1yjGaMXJsZNSzJuHS"
        )

    elif args.split == "warm_up_testing":
        data_split = DataSplit(
            GDRIVE_IDS_MVL_DATA_FN=os.path.join(
                GDRIVE_DIR, "gdrive_ids__warm_up_testing_set.csv"
            ),
            GDRIVE_IDS_LABELS_FN="",
            TYPE="warm_up_testing_set",
            GT_LABELS=False,
            GDRIVE_ID="1IE1Z7SzlQXMe9lg0CSfsVLozPvAXOJJ-",
            GDRIVE_ID_LABELS=""
        )

    elif args.split == "warm_up_training":
        data_split = DataSplit(
            # GDRIVE_IDS_MVL_DATA_FN=os.path.join(GDRIVE_DIR, 'gdrive_ids__warm_up_training_set.csv'),
            GDRIVE_IDS_MVL_DATA_FN=os.path.join(
                # GDRIVE_DIR, "gdrive_ids__warm_up_training_set_folders.csv"
                GDRIVE_DIR, "gdrive_ids__warm_up_training_set.csv"

            ),
            GDRIVE_IDS_LABELS_FN="",
            TYPE="warm_up_training_set",
            GT_LABELS=False,
            GDRIVE_ID="19rQ3YrhHHYGiDSjeGp2wFKyD8df7py90",
            GDRIVE_ID_LABELS=""
        )
        # download_data_split_by_folders(args, data_split)
        # return
    elif args.split == "challenge_training":
        data_split = DataSplit(
            # GDRIVE_IDS_MVL_DATA_FN=os.path.join(GDRIVE_DIR, 'gdrive_ids__warm_up_training_set.csv'),
            GDRIVE_IDS_MVL_DATA_FN=os.path.join(
                # GDRIVE_DIR, "gdrive_ids__warm_up_training_set_folders.csv"
                GDRIVE_DIR, "gdrive_ids__challenge_phase_training_set.csv"

            ),
            GDRIVE_IDS_LABELS_FN="",
            TYPE="challenge_phase__training_set",
            GT_LABELS=False,
            GDRIVE_ID="1bnTTzTsc547DVRSccDQCr16LWwdzXkbG",
            GDRIVE_ID_LABELS=""
        )
    elif args.split == "challenge_testing":
        data_split = DataSplit(
            # GDRIVE_IDS_MVL_DATA_FN=os.path.join(GDRIVE_DIR, 'gdrive_ids__warm_up_training_set.csv'),
            GDRIVE_IDS_MVL_DATA_FN=os.path.join(
                # GDRIVE_DIR, "gdrive_ids__warm_up_training_set_folders.csv"
                GDRIVE_DIR, "gdrive_ids__challenge_phase_testing_set.csv"

            ),
            GDRIVE_IDS_LABELS_FN="",
            TYPE="challenge_phase__testing_set",
            GT_LABELS=False,
            GDRIVE_ID="1VoTifgI8_sIfN324UOtlgB16jxHybKYC",
            GDRIVE_ID_LABELS=""
        )
    else:
        raise ValueError(f"Not implemented split: {args.split}")

    download_data_split(args, data_split)


def get_argparse():
    desc = "This script helps you to automatically download mvl-dataset in a passed output dir."

    parser = argparse.ArgumentParser(description=desc, epilog=EPILOG)

    parser.add_argument(
        "-o",
        "--output_dir",
        default=f"{DEFAULT_DOWNLOAD_DIR}",
        type=str,
        help=f"Output directory by default it will store at {DEFAULT_DOWNLOAD_DIR}.",
    )

    parser.add_argument(
        "-split",
        default="challenge_training",
        type=str,
        help="Defines the split data you want to download. Options: 'pilot', 'warm_up_testing', 'warm_up_training', 'challenge_testing', 'challenge_training' ",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_argparse()
    main(args)
