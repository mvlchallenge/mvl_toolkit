from dataclasses import dataclass
import os
import subprocess
import argparse
from mvl_challenge import ASSETS_DIR, DATA_DIR, ROOT_DIR, EPILOG, CFG_DIR
from mvl_challenge.config.cfg import get_empty_cfg, read_omega_cfg
from mvl_challenge.remote_data.download_mvl_data import download_scenes, download_google_drive_link
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
   
def download_data_split(args, data_split:DataSplit):
    #! Downloading mvl data
    zip_dir = os.path.join(args.output_dir, "zips", data_split.TYPE)
    download(data_split.GDRIVE_IDS_MVL_DATA_FN, zip_dir)
    
    # ! Unzipping mvl-data
    output_dir = os.path.join(args.output_dir, 'mvl_data')
    unzip(zip_dir, output_dir)
    
    if data_split.GT_LABELS:
        zip_dir = os.path.join(args.output_dir, "zips_labels", data_split.TYPE)
        download(data_split.GDRIVE_IDS_LABELS_FN, zip_dir)
        
        output_dir = os.path.join(args.output_dir, 'mvl_data')
        unzip(zip_dir, output_dir)
        
            
    print(f"** \tzip dir for {data_split.TYPE}:\t\t{zip_dir}")
    print(f"** \tmvl dir for {data_split.TYPE}:\t\t{output_dir}")
    print(f"*\t->>>\t{data_split.TYPE} downloaded successfully\t<<<-\t*")


def unzip(zip_dir, output_dir):
    create_directory(output_dir, delete_prev=False)
    subprocess.run(["bash", f"{ROOT_DIR}/remote_data/unzip_data.sh", 
                    "-d", f"{zip_dir}", "-o", f"{output_dir}"])

def download(gdrive_fn, zip_dir):
    create_directory(zip_dir, delete_prev=False)
    cfg = get_empty_cfg()
    cfg.output_dir = zip_dir
    cfg.ids_file = os.path.join(DATA_DIR, gdrive_fn)
    download_scenes(cfg)
    
def main(args):
    if args.split == 'pilot':
        data_split = DataSplit(
            GDRIVE_IDS_MVL_DATA_FN=os.path.join(DATA_DIR, 'gdrive_ids__pilot_set.csv'),
            GDRIVE_IDS_LABELS_FN=os.path.join(DATA_DIR, 'gdrive_ids__pilot_labels.csv'),
            TYPE='pilot_set',
            GT_LABELS=True
            
        )
    elif args.split == 'warm_up_testing':
        data_split = DataSplit(
            GDRIVE_IDS_MVL_DATA_FN=os.path.join(DATA_DIR, 'gdrive_ids__warm_up_testing_set.csv'),
            GDRIVE_IDS_LABELS_FN="",
            TYPE='warm_up_testing_set',
            GT_LABELS=False
            
        )
    elif args.split == 'warm_up_training':
            data_split = DataSplit(
            GDRIVE_IDS_MVL_DATA_FN=os.path.join(DATA_DIR, 'gdrive_ids__warm_up_training_set.csv'),
            GDRIVE_IDS_LABELS_FN="",
            TYPE='warm_up_training_set',
            GT_LABELS=False
            
        )
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
        default=f"{ASSETS_DIR}/data",
        type=str,
        help=f'Output directory by default it will store at {ASSETS_DIR}/mvl_data.'
    )

    parser.add_argument(
        '-split', 
        default="pilot",
        type=str,
        help="Defines the split data you want to download. Options: 'pilot', 'warm_up_testing', 'warm_up_training', 'challenge_testing', 'challenge_training' "
    )
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_argparse()
    main(args)
    