import os
import numpy as np
import json
from tqdm import tqdm


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
