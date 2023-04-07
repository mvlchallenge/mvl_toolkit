import argparse
import os

from mvl_challenge import DEFAULT_CKPT_DIR, GDRIVE_DIR
from mvl_challenge.remote_data.download_mvl_data import download_google_drive_link
from mvl_challenge.utils.io_utils import create_directory, read_txt_file


def main(args):

    create_directory(args.output_dir, delete_prev=False)

    gdrive_ids_fn = args.gdrive_ids
    assert os.path.exists(gdrive_ids_fn), f"Not found {gdrive_ids_fn}"

    lines = read_txt_file(gdrive_ids_fn)

    for l in lines:
        gd_id, zip_fn = [l for l in l.replace(" ", ",").split(",") if l != ""][:2]
        output_file = os.path.join(args.output_dir, zip_fn)
        download_google_drive_link(
            gd_id, output_file, f"{lines.index(l)+1}/{lines.__len__()}"
        )


def get_passed_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-f",
        "--gdrive_ids",
        type=str,
        default=f"{GDRIVE_DIR}/gdrive_ids__ckpt_hn_models.csv",
        help=f"CKPT download info.",
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=f"{DEFAULT_CKPT_DIR}",
        help=f"Default CKPT directory {DEFAULT_CKPT_DIR}.",
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_passed_args()
    main(args)
