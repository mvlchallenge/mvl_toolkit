
import argparse
import zipfile
import os 

from mp3d_fpe import create_directory, get_files_given_a_pattern


def process_arcname(list_fn):    
    common_path = os.path.commonpath(list_fn)
    return [os.path.relpath(fn, start=common_path) for fn in list_fn]

def main(args):

    # ! Create output directory
    output_dir=create_directory(args.o, delete_prev=False)    
    
    # ! Get list of files
    list_fn = get_files_given_a_pattern(
        args.d, flag_file=args.key, exclude=["depth", "vo", "hn_mp3d"], include_flag_file=True
        )

    list_arcname = process_arcname(list_fn)
    # ! Create zip files
    zip_filename = os.path.join(output_dir, f"{args.key}.zip")
    with zipfile.ZipFile(file=zip_filename, mode='w') as zf:
        [zf.write(fn, compress_type=zipfile.ZIP_STORED, arcname=arc) for fn, arc in zip(list_fn, list_arcname)]
    



def get_argparse():
    parser = argparse.ArgumentParser()
   
    parser.add_argument(
        '--key',
        # required=True,
        default="label.json",
        type=str,
        help='')

    # * Input Directory (--d)
    parser.add_argument(
        '--d',
        # required=True
        default="/media/public_dataset/MP3D_360_FPE/SINGLE_ROOM_SCENES",
        type=str,
        help='')
    
    
    # * Output Directory (--o)
    parser.add_argument(
        '--o',
        # required=True
        default="/media/public_dataset/MP3D_360_FPE/zipped_scenes2",
        type=str,
        help='')
    
    
    # * Output Directory (--o)
    parser.add_argument(
        '--unzip',
        # required=True
        default="/media/public_dataset/MP3D_360_FPE/zipped_scenes",
        type=str,
        help='')
    
    args = parser.parse_args()
    return args

    pass


if __name__ == '__main__':
    args = get_argparse()
    main(args)
    