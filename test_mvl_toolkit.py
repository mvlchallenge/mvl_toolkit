import os
import subprocess
import argparse
from mvl_challenge import (
    ASSETS_DIR,
    GDRIVE_DIR,
    ROOT_DIR,
    EPILOG,
    CFG_DIR,
    DEFAULT_DOWNLOAD_DIR,
    DEFAULT_NPZ_DIR
)
from mvl_challenge.config.cfg import get_empty_cfg, read_omega_cfg
from mvl_challenge.remote_data.download_mvl_data import (
    download_file,
    download_google_drive_link,
)
from mvl_challenge.utils.io_utils import create_directory, save_compressed_phi_coords
from mvl_challenge.datasets.mvl_dataset import MVLDataset, iter_mvl_room_scenes
from mvl_challenge.models.wrapper_horizon_net import WrapperHorizonNet
from mvl_challenge.challenge_results.create_zip_results import zip_results

GOOGLE_IDS_EXAMPLE_SCENE = "gdrive_ids__example_scene.csv"


def main(args):

    list_logs = []
    try:
        #! Downloading example scene
        list_logs.append("Download example_scene.zip")
        cfg = get_empty_cfg()
        example_scene_zip_dir = os.path.join(args.output_dir, "zips", "example_data")
        create_directory(example_scene_zip_dir, delete_prev=False)
        cfg.output_dir = example_scene_zip_dir
        cfg.ids_file = os.path.join(GDRIVE_DIR, GOOGLE_IDS_EXAMPLE_SCENE)
        download_file(cfg)

        # ! Unzipping mvl-data
        list_logs.append("Unzip example_scene.zip as mvl-data")
        cfg.output_dir = os.path.join(args.output_dir, "mvl_data")
        create_directory(cfg.output_dir, delete_prev=False)
        subprocess.run(
            [
                "bash",
                f"{ROOT_DIR}/remote_data/unzip_data.sh",
                "-d",
                f"{example_scene_zip_dir}",
                "-o",
                f"{cfg.output_dir}",
            ]
        )
        scene_list_fn = os.path.join(example_scene_zip_dir, "example_scene_list.json")
        assert os.path.exists(scene_list_fn), f"Not found {scene_list_fn}"

        #! Download CKPT
        list_logs.append("Download CKPT (Pretrained model)")
        ckpt_mp3d_id = "1W2A-_WU9d5KAwEQiTywJud2mRO3hLXqL"
        fn = os.path.join(ASSETS_DIR, "ckpt")
        fn = create_directory(fn, delete_prev=False)
        ckpt_fn = os.path.join(fn, "hn_mp3d.path")
        download_google_drive_link(ckpt_mp3d_id, ckpt_fn)

        # ! Loading List of Layout instances
        list_logs.append("Load mvl-data as a set of list Layouts")
        cfg_mvl = read_omega_cfg(f"{CFG_DIR}/eval_mvl_dataset.yaml")
        cfg_mvl.scene_dir = cfg.output_dir
        cfg_mvl.scene_list = scene_list_fn
        mvl = MVLDataset(cfg_mvl)
        [ly for ly in mvl.iter_list_ly()]

        # ! Evaluating HorizonNet
        list_logs.append("Eval list Layouts using HorizonNet")
        cfg_mvl.ckpt = ckpt_fn
        cfg_mvl.cuda = 0
        hn = WrapperHorizonNet(cfg_mvl)
        list_ly = [ly for ly in iter_mvl_room_scenes(model=hn, dataset=mvl)][0]

        # ! Saving npz estimations
        list_logs.append("Save *.npz estimations")
        npz_estimations_dir = os.path.join(DEFAULT_NPZ_DIR, "example_data")
        create_directory(npz_estimations_dir, delete_prev=False)
        for ly in list_ly:
            fn = os.path.join(npz_estimations_dir, ly.idx)
            # ! IMPORTANT: Use ALWAYS save_compressed_phi_coords()
            save_compressed_phi_coords(ly.phi_coords, fn)

        # ! Zip npz estimations
        list_logs.append("Zip *.npz estimations")
        cfg.results_dir = npz_estimations_dir
        cfg.scene_list = scene_list_fn
        zip_results(cfg)

    except Exception as err:
        [print(f"[PASSED]:\t{log}") for log in list_logs[:-1]]
        print(f"[FAILED]:\t{list_logs[-1]}")
        print(type(err))  # the exception instance
        print(err.args)  # arguments stored in .args
        print(err)
        return

    [print(f"[PASSED]:\t{log}") for log in list_logs]

    print(f"*\t->>>\tSuccessfully test mvl-toolkit\t<<<-\t*")


def get_argparse():
    desc = "This script test automatically the present mvl-toolkit using a predefined example mvl-scene."

    parser = argparse.ArgumentParser(description=desc, epilog=EPILOG)

    parser.add_argument(
        "-o",
        "--output_dir",
        default=f"{DEFAULT_DOWNLOAD_DIR}",
        type=str,
        help=f"Output directory by default it will store at {DEFAULT_DOWNLOAD_DIR}.",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_argparse()
    main(args)
