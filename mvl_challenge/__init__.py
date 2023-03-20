import os

ROOT_DIR = os.path.dirname(__file__)
CFG_DIR = os.path.join(ROOT_DIR, "data/configs")
ASSETS_DIR = os.path.join(ROOT_DIR, 'assets')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
GDRIVE_DIR = os.path.join(ROOT_DIR, 'data/gdrive_files')
SCENE_LIST_DIR = os.path.join(ROOT_DIR, 'data/scene_list')
DEFAULT_DOWNLOAD_DIR = os.path.join(ASSETS_DIR, 'data')
DEFAULT_MVL_DIR = os.path.join(ASSETS_DIR, 'data/mvl_data')
DEFAULT_NPZ_DIR = os.path.join(ASSETS_DIR, 'npz')

EPILOG = "\t * MVL-Challenge - CVPR 2023 - OmniCV workshop"
