import os
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
from mvl_challenge.config.cfg import read_omega_cfg
from mvl_challenge.models.wrapper_horizon_net import WrapperHorizonNet
from mvl_challenge import (
    ASSETS_DIR,
    DEFAULT_MVL_DIR,
    SCENE_LIST_DIR,
    DEFAULT_TRAINING_DIR
)

HN_TUTORIAL_DIR=os.path.dirname(__file__)

def get_cfg_from_args(args):
    cfg = read_omega_cfg(args.cfg)
    cfg.mvl_dir = args.scene_dir
    cfg.pilot_scene_list = args.pilot_scene_list
    cfg.output_dir = args.output_dir
    cfg.ckpt = args.ckpt
    cfg.cuda_device = args.cuda_device
    cfg.id_exp = f"{Path(cfg.ckpt).stem}__{Path(args.pilot_scene_list).stem}"
    return cfg

def main(args):
    # ! Reading configuration
    cfg = get_cfg_from_args(args)
    
    model = WrapperHorizonNet(cfg)
    
    model.prepare_for_training()
    model.set_valid_dataloader()
    model.valid_iou_loop()
    model.save_current_scores()
    while model.is_training:
        model.train_loop()
        model.valid_iou_loop()
        model.save_current_scores()
        
def get_passed_args():
    parser = argparse.ArgumentParser()
    
    default_cfg = f"{HN_TUTORIAL_DIR}/train_hn.yaml" 
    parser.add_argument(
        '--cfg',
        default=default_cfg,
        help=f'Config File. Default {default_cfg}')
    
    parser.add_argument(
        "--pilot_scene_list",
        type=str,
        default=f"{SCENE_LIST_DIR}/scene_list__warm_up_pilot_set.json",
        help="Pilot scene list",
    )
    
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=f"{DEFAULT_TRAINING_DIR}",
        help="MVL dataset directory.",
    )
    
    parser.add_argument(
        "-d",
        "--scene_dir",
        type=str,
        default=f"{DEFAULT_MVL_DIR}",
        help="MVL dataset directory.",
    )
    
    parser.add_argument(
        "--ckpt",
        default=f"{ASSETS_DIR}/ckpt/hn_mp3d.pth",
        help=f"Path to ckpt pretrained model (Default: {ASSETS_DIR}/ckpt/hn_mp3d.pth)",
    )
    
    parser.add_argument("--cuda_device", default=0, type=int, help="Cuda device. (Default: 0)")

    args = parser.parse_args()
    return args
         
if __name__ == "__main__":
    args = get_passed_args()
    main(args)
    