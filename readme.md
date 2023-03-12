## Comming soon. For more information see [mvl-challege](https://mvlchallenge.github.io/)

The tentative dates for this challenge are described as follows: 
* Warm-up Phase Open - March 15, 2023
* Challenge Phase Open - April 20, 2023
* Challenge Phase Deadline - June 1, 2023
* Winner notification - June 6, 2023

## Installation

This is only for the current branch
```bash
git clone https://github.com/mvlchallenge/mvl_toolkit.git
git branch -f mvl_chellenge_dev origin/mvl_chellenge_dev
git checkout mvl_chellenge_dev
git submodule update --init --recursive
```
You should see the `HorizonNet` folder in under `mvl_challenge/models`.

## Environment

### 1. Prepare conda env
```bash
conda create -n mvl-challenge python=3.7
```

### 2. Install mvl-toolkit
```bash
pip install -e .
```

## Usage

### 1. Download the dataset
The csv files under `mvl_challenge/data/mp3d_fpe/` specify the ID of each data. We will use these IDs to download the data from Google Drive and store them into the output directory you assign.

For example:
```bash
# You need to assign an OUTPUT_DIR
# You can change the name of the csv file to download either training or testing data
python mvl_challenge/remote_data/download_mvl_data.py -o $OUTPUT_DIR -f mvl_challenge/data/mp3d_fpe/test__google_ids__mvl_data.csv
```
The downloaded data will be in zip format, now we need to unzip them:
```bash
# You need to assign a DATASET_DIR
bash mvl_challenge/remote_data/unzip_data.sh -d $ZIP_FILE_DIR -o $DATASET_DIR
```
The expected dataset structure:   

|- DATASET_DIR  
&emsp;|- geometry_info  
&emsp;|- img  

### 2. Load data
The json files under `mvl_challenge/data/mp3d_fpe/` specify
the list of scenes and frames in the dataset.

Now we can load the data that we just downloaded and visualize them.  
For exmaple:
```bash
# DATASET_DIR is the directory where you just downloaded the data
# You can change the name of the json file to load either training or testing data
python mvl_challenge/mvl_data/load_mvl_dataset.py -d $DATASET_DIR -f mvl_challenge/data/mp3d_fpe/test__mp3d_fpe__scene_list.json
```
While running the program, you should see a sequence of panorama images (i.e., the dataset images) showing up on the screen.

### 3. Evaluate
Now we can load the data and evaluate them by the pre-trained model of HorizonNet.
```bash
# You can use -h to see more information
python mvl_challenge/mvl_data/load_and_eval_mvl_dataset.py -d $DATASET_DIR -f mvl_challenge/data/mp3d_fpe/test__mp3d_fpe__scene_list.json
```
While running the program, you should see a sequence of panorama images with the layout prediction predicted by HorizonNet showing up on the screen.

Moreover, in the end, it will pop out a window showing up the point cloud of all the layout estimation in 3D.