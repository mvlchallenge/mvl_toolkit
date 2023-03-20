import argparse
from mvl_challenge import DATA_DIR, ROOT_DIR, CFG_DIR, EPILOG, ASSETS_DIR, DEFAULT_NPZ_DIR, SCENE_LIST_DIR
from mvl_challenge.config.cfg import read_omega_cfg
from mvl_challenge.datasets.mvl_dataset import MVLDataset, iter_mvl_room_scenes
from mvl_challenge.utils.vispy_utils import plot_list_ly
from mvl_challenge.utils.image_utils import draw_boundaries_phi_coords
from imageio import imwrite
from mvl_challenge.models.wrapper_horizon_net import WrapperHorizonNet
from mvl_challenge.utils.io_utils import create_directory, process_arcname, get_scene_room_from_scene_room_idx, save_compressed_phi_coords
from mvl_challenge.utils.image_utils import plot_image
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import zipfile
import json


def zip_results(args):
    list_fn = os.listdir(args.results_dir)
    # ! if scene_list is passed
    if os.path.exists(args.scene_list):
        data_scene = json.load(open(args.scene_list, 'r'))
        scene_list = [sc for sc in data_scene.values()]
        scene_list = [item for sublist in scene_list for item in sublist]
        results_fn = [os.path.join(args.results_dir, r) for r in list_fn
                      if Path(r).stem in scene_list]
    else:
        results_fn = [os.path.join(args.results_dir, r) for r in list_fn]

    output_dir = Path(args.results_dir).parent
    results_name = Path(args.results_dir).stem
    zip_results_fn = os.path.join(output_dir, f"{results_name}.zip")
    with zipfile.ZipFile(file=zip_results_fn, mode='w') as zf:
        list_arc_fn = process_arcname(results_fn, args.results_dir)
        [(print(f"zipping {fn}"),
          zf.write(os.path.join(args.results_dir, fn),
                   compress_type=zipfile.ZIP_STORED,
                   arcname=fn))
         for fn in tqdm(list_arc_fn)
         ]


def get_argparse():
    desc = "This script create a results.zip file from a directory of npz files (estimation files), which can be submitted to EvalAi. "

    parser = argparse.ArgumentParser(
        description=desc,
        epilog=EPILOG
    )

    parser.add_argument(
        '-d', '--results_dir',
        type=str,
        default=f'{DEFAULT_NPZ_DIR}/scene_list__warm_up_pilot_set',
        help='Results directory where *.npz files were stored.'
    )

    parser.add_argument(
        "-f", "--scene_list",
        default=f"{SCENE_LIST_DIR}/scene_list__warm_up_pilot_set.json",
        help="JSON scene_list file of mvl scenes in scene_room_idx format. " +
        "If no scene_list is passed, then all npz files in results_dir will be used."
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_argparse()
    zip_results(args)
