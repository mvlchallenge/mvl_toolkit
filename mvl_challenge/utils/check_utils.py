import os
import numpy as np
import json
from tqdm import tqdm
from mvl_challenge.utils.io_utils import get_all_frames_from_scene_list


def check_geo_info_and_img(mvl_dir, list_scenes):
    """
    Checks whether the passed list scene is define in both 
    geometry info and img directory. 

    Args:
        mvl_dir (path): Path to a mvl dataset
        list_scenes (list): list if scene in scene_room_idx format

    Returns:
        Bool: passed or not the checking. 
    """

    geo_info_fn = [os.path.join(mvl_dir, 'geometry_info', f"{sc}.json") for sc in list_scenes]
    img_fn = [os.path.join(mvl_dir, 'img', f"{sc}.jpg") for sc in list_scenes]

    geo_info_check = [os.path.exists(fn) for fn in geo_info_fn]
    img_fn_check = [os.path.exists(fn) for fn in img_fn]

    output = True
    if np.sum(geo_info_check) != geo_info_check.__len__():
        [print(f"Not found {fn}") for ckpt, fn in zip(geo_info_check, geo_info_fn)
         if not ckpt]
        output = False

    if np.sum(img_fn_check) != img_fn_check.__len__():
        [print(f"Not found {fn}") for ckpt, fn in zip(img_fn_check, img_fn)
         if not ckpt]
        output = False

    return output


def check_mvl_dataset(args):
    """Checks a MVL dataset directory based on a list of scenes defined in scene_room_idx
    Args:
        args (parseargs): args.scene_dir, args.scene_list are need
    Returns:
        [dict, bool]: data_scene_dict, True or False
    """

    data_scene = json.load(open(args.scene_list, 'r'))
    check = [check_geo_info_and_img(args.scene_dir, scene_list)
             for scene_list in tqdm(data_scene.values(), desc=f"loading and checking {args.scene_dir}")
             ]
    return data_scene, np.sum(check) == check.__len__()


def check_scene_list(args):
    data = json.load(open(args.scene_list, "r"))
    list_frames = [f for f in data.values()]
    list_frames = [item for sublist in list_frames for item in sublist]
    
    list_imgs = [os.path.join(args.data_dir, 'img', f"{fn}.jpg") for fn in list_frames]
    list_geom_info = [os.path.join(args.data_dir, 'geometry_info', f"{fn}.json") for fn in list_frames]
    list_labels = [os.path.join(args.data_dir, 'labels', 'gt', f"{fn}.npz") for fn in list_frames]
    
    print(f" - Scene list: {args.scene_list} - total rooms:{list(data.keys()).__len__()} - total frames:{list_frames.__len__()}")
    if np.sum([os.path.isfile(fn) for fn in list_imgs]) == list_imgs.__len__():
        print(f" - * check images:\tPASSED")
    else:
        print(f" - * check images:\tFAILED")
    
    if np.sum([os.path.isfile(fn) for fn in list_geom_info]) == list_imgs.__len__():
        print(f" - * check geometry_info:\tPASSED")
    else:
        print(f" - * check geometry_info:\tFAILED")
     
    if np.sum([os.path.isfile(fn) for fn in list_labels]) == list_imgs.__len__():
        print(f" - * check labels:\tPASSED")
    else:
        print(f" - * check labels:\tFAILED")
        
    