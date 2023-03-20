from dataclasses import dataclass
import os
import subprocess
import argparse
from mvl_challenge import ASSETS_DIR, ROOT_DIR, EPILOG, CFG_DIR, GDRIVE_DIR, DEFAULT_MVL_DIR
from mvl_challenge.config.cfg import get_empty_cfg, read_omega_cfg
from mvl_challenge.remote_data.download_mvl_data import download_file, download_dirs
from mvl_challenge.utils.io_utils import create_directory, save_compressed_phi_coords
from mvl_challenge.datasets.mvl_dataset import MVLDataset, iter_mvl_room_scenes
from mvl_challenge.models.wrapper_horizon_net import WrapperHorizonNet
from mvl_challenge.challenge_results.create_zip_results import zip_results

@dataclass
class DataSplit:
    GDRIVE_IDS_MVL_DATA_FN: str
    GDRIVE_IDS_LABELS_FN: str
    TYPE: str
    GT_LABELS: bool
    
def download_data_split_by_folders(args, data_split:DataSplit):
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
    output_dir = os.path.join(args.output_dir, 'mvl_data')
    unzip(zip_dir, output_dir)
    
    
def download_data_split(args, data_split:DataSplit):
    #! Downloading mvl data
    zip_dir = os.path.join(args.output_dir, "zips", data_split.TYPE)
    download_gdrive_file(data_split.GDRIVE_IDS_MVL_DATA_FN, zip_dir)
    
    # ! Unzipping mvl-data
    output_dir = os.path.join(args.output_dir, 'mvl_data')
    unzip(zip_dir, output_dir)
    
    if data_split.GT_LABELS:
        zip_dir = os.path.join(args.output_dir, "zips_labels", data_split.TYPE)
        download_gdrive_file(data_split.GDRIVE_IDS_LABELS_FN, zip_dir)
        
        output_dir = os.path.join(args.output_dir, 'mvl_data')
        unzip(zip_dir, output_dir)
        
            
    print(f"** \tzip dir for {data_split.TYPE}:\t\t{zip_dir}")
    print(f"** \tmvl dir for {data_split.TYPE}:\t\t{output_dir}")
    print(f"*\t->>>\t{data_split.TYPE} downloaded successfully\t<<<-\t*")


def unzip(zip_dir, output_dir):
    create_directory(output_dir, delete_prev=False)
    subprocess.run(["bash", f"{ROOT_DIR}/remote_data/unzip_data.sh", 
                    "-d", f"{zip_dir}", "-o", f"{output_dir}"])

def download_gdrive_file(gdrive_fn, zip_dir):
    create_directory(zip_dir, delete_prev=False)
    cfg = get_empty_cfg()
    cfg.output_dir = zip_dir
    cfg.ids_file = os.path.join(GDRIVE_DIR, gdrive_fn)
    download_file(cfg)


def main(args):
    if args.split == 'pilot':
        data_split = DataSplit(
            GDRIVE_IDS_MVL_DATA_FN=os.path.join(GDRIVE_DIR, 'gdrive_ids__pilot_set.csv'),
            GDRIVE_IDS_LABELS_FN=os.path.join(GDRIVE_DIR, 'gdrive_ids__pilot_labels.csv'),
            TYPE='pilot_set',
            GT_LABELS=True
        )
        
    elif args.split == 'warm_up_testing':
        data_split = DataSplit(
            GDRIVE_IDS_MVL_DATA_FN=os.path.join(GDRIVE_DIR, 'gdrive_ids__warm_up_testing_set.csv'),
            GDRIVE_IDS_LABELS_FN="",
            TYPE='warm_up_testing_set',
            GT_LABELS=False
        )
        
    elif args.split == 'warm_up_training':
            data_split = DataSplit(
            # GDRIVE_IDS_MVL_DATA_FN=os.path.join(GDRIVE_DIR, 'gdrive_ids__warm_up_training_set.csv'),
            GDRIVE_IDS_MVL_DATA_FN=os.path.join(GDRIVE_DIR, 'gdrive_ids__warm_up_training_set_folders.csv'),
            GDRIVE_IDS_LABELS_FN="",
            TYPE='warm_up_training_set',
            GT_LABELS=False
        )
            download_data_split_by_folders(args, data_split)
            return
    else:
        raise ValueError(f"Not implemented split: {args.split}")
        
    download_data_split(args, data_split)

    
def get_argparse():
    desc = "This script helps you to automatically download mvl-dataset in a passed output dir."

    parser = argparse.ArgumentParser(
        description=desc,
        epilog=EPILOG
    )

    parser.add_argument(
        '-o', '--output_dir',
        default=f"{DEFAULT_MVL_DIR}",
        type=str,
        help=f'Output directory by default it will store at {DEFAULT_MVL_DIR}.'
    )

    parser.add_argument(
        '-split', 
        default="warm_up_training",
        type=str,
        help="Defines the split data you want to download. Options: 'pilot', 'warm_up_testing', 'warm_up_training', 'challenge_testing', 'challenge_training' "
    )
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_argparse()
    main(args)
    