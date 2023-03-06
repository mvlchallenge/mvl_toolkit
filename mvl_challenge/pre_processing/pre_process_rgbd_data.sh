#!/bin/sh
OUTPUT_DIR=/media/public_dataset/mvl_challenge
# ! scene_room_idx files 
python mvl_challenge/pre_processing/create_scene_room_idx_list.py \
        -d /media/public_dataset/HM3D-MVL/test/ \
        -f hm3d_mvl__scene_test_list \
        -o "$OUTPUT_DIR"/ -m 5

python mvl_challenge/pre_processing/create_scene_room_idx_list.py \
        -d /media/public_dataset/MP3D_360_FPE/SINGLE_ROOM_SCENES/ \
        -f mp3d_fpe__single_room_scene_list \
        -o "$OUTPUT_DIR"/ -m 5

python mvl_challenge/pre_processing/create_scene_room_idx_list.py \
        -d /media/public_dataset/MP3D_360_FPE/MULTI_ROOM_SCENES/ \
        -f mp3d_fpe__multi_room_scene_list \
        -o "$OUTPUT_DIR"/ -m 5

# # ! geometry info files
python mvl_challenge/pre_processing/create_geometry_info_files.py \
        -d /media/public_dataset/HM3D-MVL/test/ \
        --scene_list "$OUTPUT_DIR"/hm3d_mvl__scene_test_list.json \
        -o "$OUTPUT_DIR"/hm3d_mvl/geometry_info

python mvl_challenge/pre_processing/create_geometry_info_files.py \
        -d /media/public_dataset/MP3D_360_FPE/SINGLE_ROOM_SCENES/ \
        --scene_list "$OUTPUT_DIR"/mp3d_fpe__single_room_scene_list.json \
        -o "$OUTPUT_DIR"/mp3d_fpe/geometry_info

python mvl_challenge/pre_processing/create_geometry_info_files.py \
        -d /media/public_dataset/MP3D_360_FPE/MULTI_ROOM_SCENES/ \
        --scene_list "$OUTPUT_DIR"/mp3d_fpe__multi_room_scene_list.json \
        -o "$OUTPUT_DIR"/mp3d_fpe/geometry_info

# ! RGB files
python mvl_challenge/pre_processing/create_rgb_files.py \
        -d /media/public_dataset/HM3D-MVL/test/ \
        -g "$OUTPUT_DIR"/hm3d_mvl/geometry_info \
        -o "$OUTPUT_DIR"/hm3d_mvl/img

python mvl_challenge/pre_processing/create_rgb_files.py \
        -d /media/public_dataset/MP3D_360_FPE/SINGLE_ROOM_SCENES/ \
        -g "$OUTPUT_DIR"/mp3d_fpe/geometry_info \
        -o "$OUTPUT_DIR"/mp3d_fpe/img

python mvl_challenge/pre_processing/create_rgb_files.py \
        -d /media/public_dataset/MP3D_360_FPE/MULTI_ROOM_SCENES/ \
        -g "$OUTPUT_DIR"/mp3d_fpe/geometry_info \
        -o "$OUTPUT_DIR"/mp3d_fpe/img