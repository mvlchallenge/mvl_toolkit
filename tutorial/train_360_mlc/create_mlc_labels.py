import os
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from imageio import imwrite
from mlc.mlc import compute_pseudo_labels
from mvl_challenge.datasets.mvl_dataset import iter_mvl_room_scenes
from mvl_challenge.config.cfg import read_omega_cfg
from mvl_challenge import DEFAULT_MVL_DIR
from mvl_challenge.utils.io_utils import create_directory, save_compressed_phi_coords
from mvl_challenge.datasets.mvl_dataset import MVLDataset
from mvl_challenge.models.wrapper_horizon_net import WrapperHorizonNet
from mvl_challenge.utils.image_utils import (
    draw_boundaries_uv, 
    draw_uncertainty_map, 
    COLOR_MAGENTA)
from mvl_challenge import (
    ASSETS_DIR,
    DEFAULT_MVL_DIR,
    SCENE_LIST_DIR,
)

MLC_TUTORIAL_DIR=os.path.dirname(__file__)

def create_mlc_label_dirs(cfg):
    """
    Create the directories and create the cfg.mlc_dir into the cfg class used for initialization
    """
    create_directory(os.path.join(cfg.output_dir, cfg.id_exp), delete_prev=True)
    # ! Initializing cfg.mlc_dir class directory
    cfg.mlc_dir = dict(
        phi_coords = os.path.join(cfg.output_dir, cfg.id_exp, "mlc_label"),
        std = os.path.join(cfg.output_dir, cfg.id_exp, "std"),
        vis = os.path.join(cfg.output_dir, cfg.id_exp, "mlc_vis"))

    create_directory(cfg.mlc_dir.phi_coords, delete_prev=True)
    create_directory(cfg.mlc_dir.std, delete_prev=True)
    create_directory(cfg.mlc_dir.vis, delete_prev=True)

def save_visualization(fn, img_boundaries, sigma_map):
    plt.figure(0, dpi=200)
    plt.clf()
    plt.subplot(211)
    plt.suptitle(Path(fn).stem)
    plt.imshow(img_boundaries)
    plt.axis('off')
    plt.subplot(212)
    plt.imshow(sigma_map)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    plt.savefig(fn,bbox_inches='tight')
    plt.close()

def compute_and_save_mlc_labels(list_ly):
    for ref in tqdm(list_ly, desc="Estimating MLC Labels"):
        uv_ceiling_ps, uv_floor_ps, std_ceiling, std_floor, _ = compute_pseudo_labels(
            list_frames=list_ly,
            ref_frame=ref,
        )

        # ! Saving pseudo labels
        uv = np.vstack((uv_ceiling_ps[1], uv_floor_ps[1]))
        std = np.vstack((std_ceiling, std_floor))
        phi_coords = (uv / 512 + 0.5) * np.pi
        
        # ! NOTE: 360-MLC expects npy files as pseudo labels
        fn = os.path.join(ref.cfg.mlc_dir.phi_coords, f"{ref.idx}")
        np.save(fn, phi_coords)
        fn = os.path.join(ref.cfg.mlc_dir.std, f"{ref.idx}")
        np.save(fn, std)
        
        img = ref.get_rgb()
        draw_boundaries_uv(
            image=img,
            boundary_uv=uv_ceiling_ps,
            color=COLOR_MAGENTA
        )
        
        draw_boundaries_uv(
            image=img,
            boundary_uv=uv_floor_ps,
            color=COLOR_MAGENTA
        )
        
        sigma_map = draw_uncertainty_map(
            peak_boundary=np.hstack((uv_ceiling_ps, uv_floor_ps)),
            sigma_boundary=np.hstack((std_ceiling, std_floor))
        )
        
        fn = os.path.join(ref.cfg.mlc_dir.vis, f"{ref.idx}.jpg")
        save_visualization(fn, img, sigma_map)
        
def get_cfg_from_args(args):
    cfg = read_omega_cfg(args.cfg)
    cfg.mvl_dir = args.scene_dir
    cfg.scene_list = args.scene_list
    cfg.output_dir = args.output_dir
    cfg.ckpt = args.ckpt
    cfg.cuda_device = args.cuda_device
    cfg.id_exp = f"mlc__{Path(cfg.ckpt).stem}__{Path(args.scene_list).stem}"
    return cfg
 
def main(args):
    # ! Reading configuration
    cfg = get_cfg_from_args(args)
    
    mvl = MVLDataset(cfg)
    hn = WrapperHorizonNet(cfg)

    mvl.print_mvl_data_info()
    create_mlc_label_dirs(cfg)
    
    for list_ly in iter_mvl_room_scenes(model=hn, dataset=mvl):
        compute_and_save_mlc_labels(list_ly)

def get_passed_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--cfg',
        default=f"{MLC_TUTORIAL_DIR}/create_mlc_labels.yaml",
        help=f'Config File. Default {MLC_TUTORIAL_DIR}/create_mlc_labels.yaml')
    
    parser.add_argument(
        "-f",
        "--scene_list",
        type=str,
        default=f"{SCENE_LIST_DIR}/scene_list__warm_up_training_set.json",
        help="Scene_list of mvl scenes in scene_room_idx format.",
    )
    
    parser.add_argument(
        "-d",
        "--scene_dir",
        type=str,
        default=f"{DEFAULT_MVL_DIR}",
        help="MVL dataset directory.",
    )
    
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=f"{DEFAULT_MVL_DIR}/labels",
        help="MVL dataset directory.",
    )
    
    parser.add_argument(
        "--ckpt",
        default=f"{ASSETS_DIR}/ckpt/hn_mp3d.pth",
        help="Path to ckpt pretrained model (Default: mp3d)",
    )
    
    parser.add_argument("--cuda_device", default=0, type=int, help="Cuda device. (Default: 0)")

    args = parser.parse_args()
    return args
         
if __name__ == "__main__":
    args = get_passed_args()
    main(args)
    