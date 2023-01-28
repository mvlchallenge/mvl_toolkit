import argparse
import os
import zipfile

from tqdm import tqdm
from pathlib import Path

from mp3d_fpe.io_utils import create_directory, get_files_given_a_pattern


def process_arcname(list_fn):
    common_path = os.path.commonpath(list_fn)
    return [os.path.relpath(fn, start=common_path) for fn in list_fn]


def zipping(args):
    # ! Create output directory
    output_dir = create_directory(args.o, delete_prev=False)

    # ! Get list of files
    list_targets = get_files_given_a_pattern(
        Path(args.s).resolve(),
        flag_file=args.k,
        exclude=["depth", "vo", "hn_mp3d"],
        include_flag_file=True,
        isDir=args.d)
    
    list_arcname = process_arcname(list_targets)
    if args.d:
        # ! Create zip files from a directory
        for dir, arc in zip(list_targets, list_arcname):
            zip_filename = os.path.join(output_dir,
                                        f"{arc.replace('/', '_')}.zip")
            list_fn = os.listdir(dir)
            with zipfile.ZipFile(file=zip_filename, mode='w') as zf:
                [
                    zf.write(os.path.join(dir, fn),
                             compress_type=zipfile.ZIP_STORED,
                             arcname=os.path.join(arc, fn))
                    for fn in tqdm(list_fn, desc=f"Zipping {arc}")
                ]

    else:
        # ! Create zip file
        zip_filename = os.path.join(output_dir, f"{args.k}.zip")
        with zipfile.ZipFile(file=zip_filename, mode='w') as zf:
            [
                zf.write(fn, compress_type=zipfile.ZIP_STORED, arcname=arc)
                for fn, arc in zip(list_targets, list_arcname)
            ]

def unzipping(args):
    # ! Create output directory
    output_dir = create_directory(args.o, delete_prev=False)

    # ! Get list of files
    list_targets = get_files_given_a_pattern(
        Path(args.s).resolve(),
        flag_file=args.k,
        exclude=["depth", "vo", "hn_mp3d"],
        include_flag_file=True)

    import pdb; pdb.set_trace()
    for zip_fn in tqdm(list_targets, desc=f"Unzipping...{args.k}"):
        with zipfile.ZipFile(file=zip_fn, mode='r') as zf:
            zf.extractall(output_dir)
            
def get_argparse():
    parser = argparse.ArgumentParser()

    # * Key argument in source (-k)
    parser.add_argument(
        '-k',
        required=True,
        # default="label.json",
        # default="rgb",
        type=str,
        help='Key argument in source')

    # * Input Directory (-source)
    parser.add_argument(
        '-s',
        required=True,
        # default="/media/public_dataset/MP3D_360_FPE/SINGLE_ROOM_SCENES",
        type=str,
        help='Input Directory (-source)')

    # * Output Directory (-o)
    parser.add_argument(
        '-o',
        required=True,
        # default="/media/public_dataset/MP3D_360_FPE/zipped_mp3d_fpe",
        type=str,
        help='Output Directory (-o)')

    # * Is Directory (-d)
    parser.add_argument(
        '-d',
        # required=True
        action='store_true',
        help=' Is Directory (-d)?')

    # * Unzip? (-u)
    parser.add_argument(
        '-u',
        # required=True
        action='store_true',
        help=' Unzip?')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_argparse()
    if args.u:
        unzipping(args)
    else:
        zipping(args)
