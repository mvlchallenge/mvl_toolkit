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
You should see a non-empty `HorizonNet` folder under `mvl_challenge/models`.

## Environment

### 1. Prepare conda env
```bash
conda create -n mvl-challenge python=3.9
```

### 2. Install mvl-toolkit
```bash
pip install -e .
```

## Usage

### 1. Download the dataset
The csv files under `mvl_challenge/data/mp3d_fpe/` specify the ID of each data. We will use these IDs to download the data from Google Drive (zip format) and store them into `{ZIP_DIR}`.

For example:
```bash
# You can change the csv file to download different data
python mvl_challenge/remote_data/download_mvl_data.py -o {ZIP_DIR} -f mvl_challenge/data/mp3d_fpe/test__google_ids__mvl_data.csv
```
See `python mvl_challenge/remote_data/download_mvl_data.py -h` for more detail.

Next, we will unzip the data:
```bash
bash mvl_challenge/remote_data/unzip_data.sh -d {ZIP_DIR} -o {MVL_DATA_DIR}
```
`{MVL_DATA_DIR}` is the final dataset directory storing all the data we will use in this challenge.

The expected structure:   

| - {MVL_DATA_DIR}/  
&emsp;| - geometry_info/  
&emsp;| - img/  
&emsp;| - labels/  

In all the sub-directory above, the data (file name) is in `{scene}_{version}_{room}_{frame}` format, and we call it `MVL_DATA_FORMAT`. For example, `E9uDoFAP3SH_1_room0_982`.

### 2. Load data
`mvl_challenge/data/mp3d_fpe/test__{type}__scene_list.json` represent the `scene_list` which lists the data in `MVL_DATA_FORMAT`. For example, `test__gt_labels__scene_list.json` lists all the data of ground truth labels we have in `MVL_DATA_FORMAT`.

Now we can load the data that we just downloaded and visualize them.  
For exmaple:
```bash
# You can change the json file to load different data
python mvl_challenge/mvl_data/load_mvl_dataset.py -d {MVL_DATA_DIR} -f mvl_challenge/data/mp3d_fpe/test__gt_labels__scene_list.json
```

See `python mvl_challenge/mvl_data/load_mvl_dataset.py -h` for more detail.

While running the program, you should see a sequence of panorama images (i.e., the dataset images) showing up on the screen.

### 3. Evaluate
Now we can load the data and evaluate them by the pre-trained model of HorizonNet.
```bash
python mvl_challenge/mvl_data/load_and_eval_mvl_dataset.py -d {MVL_DATA_DIR} -f mvl_challenge/data/mp3d_fpe/test__mp3d_fpe__scene_list.json
```

See `mvl_challenge/mvl_data/load_and_eval_mvl_dataset.py -h` for more detail.

While running the program, you should see a sequence of panorama images with the layout prediction predicted by HorizonNet showing up on the screen.

In the end, it will pop out a window showing the point cloud of all the layout estimation in 3D.