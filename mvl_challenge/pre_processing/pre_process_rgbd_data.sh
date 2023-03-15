#!/bin/sh

# DATA_NAME=hm3d_mvl
# RGBD_DATA_DIR=/media/public_dataset/HM3D-MVL/test
# SCENE_LIST="$DATA_NAME"__test__scene_list

DATA_NAME=mp3d_fpe
RGBD_DATA_DIR=/media/public_dataset/MP3D_360_FPE/SINGLE_ROOM_SCENES
SCENE_LIST="$DATA_NAME"__single_room__scene_list

OUTPUT_DIR=/media/public_dataset/mvl_challenge/$DATA_NAME

# ! Create scene_room_idx files --> *__scene_list.json 
python mvl_challenge/pre_processing/create_scene_room_idx_list.py \
        -d  $RGBD_DATA_DIR \
        -f  $SCENE_LIST \
        -o "$OUTPUT_DIR" -m 5

# # ! geometry info files --> scene_room_idx.json --> cam pose + camera height
# python mvl_challenge/pre_processing/create_geometry_info_files.py \
#         -d  $RGBD_DATA_DIR \
#         --scene_list "$OUTPUT_DIR"/"$SCENE_LIST".json \
#         -o "$OUTPUT_DIR"/geometry_info

# # ! RGB files
# python mvl_challenge/pre_processing/create_rgb_files.py \
#         -d  $RGBD_DATA_DIR \
#         -g "$OUTPUT_DIR"/geometry_info \
#         -o "$OUTPUT_DIR"/img

# # ! create GT labels --> scene_room_idx.npz --> phi_coords GT
# python mvl_challenge/pre_processing/create_npz_labels.py \
#         -d  $RGBD_DATA_DIR \
#         -g "$OUTPUT_DIR"/geometry_info \
#         -o "$OUTPUT_DIR"/labels/gt

# # ! Zipping mvl-data
# python mvl_challenge/remote_data/zip_mvl_dataset.py \
#         -d $OUTPUT_DIR \
#         --scene_list "$OUTPUT_DIR"/"$SCENE_LIST".json \
#         -o "$OUTPUT_DIR"/zips/mvl_data

# python mvl_challenge/remote_data/zip_mvl_dataset.py \
#         -d "$OUTPUT_DIR" \
#         --scene_list "$OUTPUT_DIR"/labels/gt_labels__scene_list.json \
#         -o "$OUTPUT_DIR"/zips/labels \
#         --labels