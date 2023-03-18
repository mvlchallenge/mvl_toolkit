import os
import json
import argparse
import numpy as np
from mvl_challenge.utils.io_utils import save_json_dict
from mvl_challenge.config.cfg import get_empty_cfg
from pathlib import Path
from mvl_challenge.scene_list__get_info import print_scene_data_info


def prune_scene_data(scene_data, max_fr):
    new_scene_data = {}
    for room_id, list_frames in scene_data.items():
        new_scene_data[room_id] = prune_list_frames(list_frames, max_fr)        
    return new_scene_data


def prune_list_frames(list_fr, max_fr):
    print(f"Initial number of frames: {list_fr.__len__()}")
    while True:
        ratio = round(list_fr.__len__()/max_fr)
        if ratio <= 1:
            break
        list_fr = [f for idx, f in enumerate(list_fr) if idx % ratio == 0]
        if list_fr.__len__() < max_fr:
            break
    list_fr.sort(key=lambda x: int(x.split("_")[-1]))
    print(f"Final number of frames: {list_fr.__len__()}")
    return list_fr

def diff_scene_lists(scene_data_1, scene_data_2):
    new_scene_data = {}
    room_ref_list = list(scene_data_1.keys())
    room_list = list(scene_data_2.keys())
    room_diff = [r for r in room_ref_list if r not in room_list]
    for room_id, scene_idx in scene_data_1.items():
        if room_id in room_diff:
            new_scene_data[room_id] = scene_idx        
    return new_scene_data

def intersect_scene_lists(scene_data_1, scene_data_2):
    new_scene_data = {}
    room_ref_list = list(scene_data_1.keys())
    room_list = list(scene_data_2.keys())
    common_rooms = [r for r in room_ref_list if r in room_list]
    for room_id, scene_idx in scene_data_1.items():
        if room_id in common_rooms:
            new_scene_data[room_id] = scene_idx        
    return new_scene_data

def complement_scene_lists(scene_data_1, scene_data_2):
    intersect__list =  intersect_scene_lists(scene_data_1, scene_data_2)
    diff_1_list = diff_scene_lists(scene_data_1, intersect__list)
    diff_2_list = diff_scene_lists(scene_data_2, intersect__list)
    new_scene_data = merge_scene_lists(diff_1_list, diff_2_list)
    return new_scene_data


def merge_scene_lists(scene_data_1, scene_data_2):
    new_scene_data = {}
    for data in (scene_data_1, scene_data_2):
        for room_id, scene_idx in data.items():
            new_scene_data[room_id] = scene_idx        
    return new_scene_data

def split_scene_list(scene_data, split_list):
    new_scene_data = {}
    for split, scenes in split_list.items():
        print(f"Split: {split}")
        new_scene_data[split] = {k: v for k, v in scene_data.items() if k in scenes}
        if list(new_scene_data[split].keys()).__len__() == 0:
            new_scene_data[split] = {k: v for k, v in scene_data.items() if k.split("_")[0] in scenes}
            if list(new_scene_data[split].keys()).__len__() == 0:
                new_scene_data[split] = {k: v for k, v in scene_data.items() if k.split("_room")[0] in scenes}
                
    return new_scene_data

def sampling_scene_list(scene_data, ratio):
    list_rooms = list(scene_data.keys())
    if ratio > 1:
        room_number = int(ratio)
    else:
        room_number = int(ratio * list_rooms.__len__())
    np.random.shuffle(list_rooms)
    return {r: scene_data[r] for r in list_rooms[:room_number]} 
    
    
    
def main(args):
    if args.r > 0: 
        scene_data = json.load(open(args.scene_list))
        new_scene_data = sampling_scene_list(scene_data, args.r)    
    elif args.action == "split": 
        scene_data = json.load(open(args.scene_list))
        split_list = json.load(open(args.x))
        new_scene_data  = split_scene_list(scene_data, split_list)
        for key, data in new_scene_data.items():
            fn = os.path.join(os.path.dirname(args.scene_list), f"{Path(args.o).stem}__{key}.json")
            save_json_dict(fn, data)
            print(f"Scene List saved as {fn}")
        return
            
    elif args.action == "merge": 
        scene_data = json.load(open(args.scene_list))
        another_scene_data = json.load(open(args.x))
        new_scene_data  = merge_scene_lists(scene_data, another_scene_data)
    elif args.action == 'diff':
        scene_data = json.load(open(args.scene_list))
        another_scene_data = json.load(open(args.x))
        new_scene_data  = diff_scene_lists(scene_data, another_scene_data)
    elif args.action == 'intersect':
        scene_data = json.load(open(args.scene_list))
        another_scene_data = json.load(open(args.x))
        new_scene_data  = intersect_scene_lists(scene_data, another_scene_data)
    elif args.action == 'complement':
        scene_data = json.load(open(args.scene_list))
        another_scene_data = json.load(open(args.x))
        new_scene_data  = complement_scene_lists(scene_data, another_scene_data)
    
    elif args.prune > 0:
        scene_data = json.load(open(args.scene_list))
        new_scene_data  = prune_scene_data(scene_data, args.prune)
    else:
        print("Not action found")
        return
    
    if args.v:
        print_scene_data_info(new_scene_data)
        return
    
    fn = os.path.join(os.path.dirname(args.scene_list), f"{Path(args.o).stem}.json")
    save_json_dict(fn, new_scene_data)
    print(f"Scene List saved as {fn}")
    
def get_argparse():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-f', '--scene_list',
        # required=True,
        # default=f"{ASSETS_DIR}/stats/hm3d_mvl__train__scene_list.json",
        # default=f"{ASSETS_DIR}/stats/mp3d_fpe__multiple_rooms__scene_list.json",
        # default="/media/public_dataset/mvl_challenge/hm3d_mvl/03.14.2023__all_hm3d_mvl__scene_list.json",
        default="/media/public_dataset/mvl_challenge/hm3d_mvl/scene_list__test.json",
        type=str,
        help='Original scene list (scene_room_idx json file).'
    )

    parser.add_argument(
        '-x', '--x',
        # required=True,
        # default=f"{ASSETS_DIR}/stats/hm3d_mvl__train__scene_list.json",
        # default=f"{ASSETS_DIR}/stats/mp3d_fpe__multiple_rooms__scene_list.json",
        # default="/media/public_dataset/mvl_challenge/hm3d_mvl/03.14.2023__all_hm3d_mvl__scene_list.json",
        default="/media/public_dataset/mvl_challenge/hm3d_mvl/scene_list__test.json",
        type=str,
        help='extra files.'
    )
    
    parser.add_argument(
        '-o',
        default="pruned_scene_list",
        type=str,
        help='Output scene_list name.'
        )

    parser.add_argument(
        '-a', "--action",
        type=str,
        help='Actions over the passed files [merge, split, diff, intersect, complement].'
    )
    
    parser.add_argument(
        '-r', 
        default=-1,
        type=float,
        help='Random sampling.'
    )
    
    parser.add_argument(
        "--prune",
        type=int,
        default=-1,
        help='Max number of fr per room.'
    )
    
    parser.add_argument(
        "-v",
        action='store_true',
        help='To visualize info without creating a file'
    )
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_argparse()
    main(args)
    