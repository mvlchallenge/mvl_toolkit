
from mvl_challenge.handle_remote_data.zip_utils import get_argparse, zip_mvl_data

if __name__ == '__main__':
    args = get_argparse()
    zip_mvl_data(args)
    
