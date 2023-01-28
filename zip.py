from mp3d_fpe.zip_utils import get_argparse, unzipping, zipping

if __name__ == '__main__':
    args = get_argparse()
    
    if args.u:
        unzipping(args)
    else:
        zipping(args)
