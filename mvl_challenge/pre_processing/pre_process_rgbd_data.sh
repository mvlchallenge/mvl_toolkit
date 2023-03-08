#!/bin/sh
OUTPUT_DIR=/media/NFS/kike/360_Challenge/mvl_toolkit/mvl_challenge/assets/mvl_data/mp3d_fpe
RGBD_DATA_DIR=/media/public_dataset/MP3D_360_FPE/SINGLE_ROOM_SCENES/
DATA_NAME=mp3d_fpe__single_room__scene_list

# RGBD_DATA_DIR=/media/public_dataset/MP3D_360_FPE/MULTI_ROOM_SCENES/
# DATA_NAME=mp3d_fpe__multi_room__scene_list


# ! scene_room_idx files 
python mvl_challenge/pre_processing/create_scene_room_idx_list.py \
        -d  $RGBD_DATA_DIR \
        -f  $DATA_NAME \
        -o "$OUTPUT_DIR" -m 5

# ! geometry info files
python mvl_challenge/pre_processing/create_geometry_info_files.py \
        -d  $RGBD_DATA_DIR \
        --scene_list "$OUTPUT_DIR"/"$DATA_NAME".json \
        -o "$OUTPUT_DIR"/geometry_info

# ! RGB files
python mvl_challenge/pre_processing/create_rgb_files.py \
        -d  $RGBD_DATA_DIR \
        -g "$OUTPUT_DIR"/geometry_info \
        -o "$OUTPUT_DIR"/img