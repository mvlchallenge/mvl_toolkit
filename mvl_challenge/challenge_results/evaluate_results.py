import argparse
from mvl_challenge import DATA_DIR, ROOT_DIR, CFG_DIR, EPILOG, ASSETS_DIR
from mvl_challenge.config.cfg import read_omega_cfg
from mvl_challenge.datasets.mvl_dataset import MVLDataset, iter_mvl_room_scenes
from mvl_challenge.utils.vispy_utils import plot_list_ly
from mvl_challenge.utils.image_utils import draw_boundaries_phi_coords
from imageio import imwrite
from mvl_challenge.models.wrapper_horizon_net import WrapperHorizonNet
from mvl_challenge.utils.io_utils import create_directory, save_json_dict, load_gt_label, get_scene_room_from_scene_room_idx
from mvl_challenge.utils.image_utils import plot_image
from mvl_challenge.utils.eval_utils import eval_2d3d_iuo
import numpy as np
import os
from pathlib import Path



def get_cfg_from_args(args):
    cfg = read_omega_cfg(args.cfg)
    cfg.scene_dir = args.scene_dir
    cfg.scene_list = args.scene_list
    cfg.ckpt = args.ckpt
    cfg.cuda = args.cuda
    return cfg


def main(args):
    cfg = get_cfg_from_args(args)
    mvl = MVLDataset(cfg)
    hn = WrapperHorizonNet(cfg)

    # ! Join the output_dir and the scene_list
    output_dir = create_directory(args.output_dir, delete_prev=False)
    results = {}
        
    for list_ly in iter_mvl_room_scenes(model=hn, dataset=mvl):
        for ly in list_ly:            
            
            phi_coords_gt = load_gt_label(os.path.join(
                args.scene_dir, "labels", "gt", f"{ly.idx}.npz"
            ))
            
            phi_coords_est = ly.phi_coords
            
            results_2d3d_iou = eval_2d3d_iuo(
                phi_coords_est=phi_coords_est,
                phi_coords_gt_bon=phi_coords_gt,
                ch=ly.camera_height
            )
            
            if np.min(results_2d3d_iou) < 0:
                # ! IoU evaluation failed, then IoU is 0 (the highest penalty) 
                results[f"{ly.idx}__2dIoU"] = 0
                results[f"{ly.idx}__3dIoU"] = 0
            
            results[f"{ly.idx}__2dIoU"] = results_2d3d_iou[0]
            results[f"{ly.idx}__3dIoU"] = results_2d3d_iou[1] 

            
    _2dIoU = np.mean([value for key, value in results.items() if "2dIoU" in key])
    _3dIoU = np.mean([value for key, value in results.items() if "3dIoU" in key])
    
    results["total__m2dIoU"] =_2dIoU
    results["total__m3dIoU"] =_3dIoU
    fn = os.path.join(output_dir, f"{Path(args.scene_list).stem}__{args.ckpt}.json")
    save_json_dict(filename=fn, dict_data=results)
     
    print(f"2d-IoU: {_2dIoU:2.3f}\t3d-IoU: {_3dIoU:2.3f}")  
    

def get_argparse():
    desc = "This script evaluates 2d-IoU, 3d-IoU from a set of estimated phi_coords. " + \
        "Note that this script assumes you have access to some GT labels. " + \
        "The passed cfg file is the yaml configuration with all hyperparameters set to default values."

    parser = argparse.ArgumentParser(
        description=desc,
        epilog=EPILOG
    )

    parser.add_argument(
        '-d', '--scene_dir',
        type=str,
        default=f'{ASSETS_DIR}/tmp/mvl_data/',
        help='MVL dataset directory.'
    )

    parser.add_argument(
        "--cfg",
        type=str,
        default=f"{CFG_DIR}/eval_mvl_dataset.yaml",
        help=f"Config file to load a MVL dataset. For this script model cfg most be defined in the cfg file too. (Default: {CFG_DIR}/eval_mvl_dataset.yaml)"
    )

    parser.add_argument(
        "-f", "--scene_list",
        type=str,
        default=f"{DATA_DIR}/mp3d_fpe/test__gt_labels__scene_list.json",
        help="Scene_list of mvl scenes in scene_room_idx format."
    )

    parser.add_argument(
        "--ckpt",
        default="mp3d",
        help="Pretrained model ckpt (Default: mp3d)"
    )

    parser.add_argument(
        "--cuda",
        default=0,
        type=int,
        help="Cuda device. (Default: 0)"
    )

    parser.add_argument(
        "-o", "--output_dir",
        # required=True,
        default=f'{ASSETS_DIR}/tmp/results/',
        help="Output directory where to store phi_coords estimations."
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_argparse()
    main(args)
