## General Information

This toolkit is part of the Multi-view Layout Estimation Challenge (mvl-challenge) of the [Omnidirectional Computer Vision (OmniCV) workshop](https://sites.google.com/view/omnicv2023/home?authuser=0) at [CVPR'23](https://cvpr2023.thecvf.com/). To participate and submit results join us on [EvalAi](https://eval.ai/web/challenges/challenge-page/1906/). 
For more information visit our official site [mvl-challege](https://sites.google.com/view/omnicv2023/challenges/multi-view-layout-challenge?authuser=0)

The tentative dates for this challenge are described as follows: 
* Warm-up Phase Open - March 20, 2023
* Challenge Phase Open - May 1, 2023
* Challenge Phase Deadline - June 1, 2023
* Winner notification - June 6, 2023

Note that this challenge proposes two phases, warm-up and challenge phases. The former plays the role of a playground stage, where participants can evaluate their models without any restriction of number of submissions. The latter aims to evaluate the parteicipant models in a selected testing set which will be released later in the challenge phase opening on May 1st, 2023.

For public queries, discussion, and free access tutorials please join us in our [Slack workspace](https://join.slack.com/t/mvl-challenge/shared_invite/zt-1m95ef0hy-ViG7fSeTt1EqiosRlZoDvQ). 

## What can you do with this toolkit?

With the aim of providing support to the participants of the mvl-challenge, the present implementation offers the following capabilities:

1. Download the training, testing and pilot datasets used in this challenge. 
2. Load and register the data as an instance `<Layout class>`, that simplifies camera and layout projection for you. 
3. Load and retrive multiple `<Layout class>` instances associated to a query room. 
5. Provide examples of how to evaluate a layout estimation using `<Layout class>` instances. 
6. Provide methods to save, zip and submit layout estimates to EvalAI. 

## Installation

1. Prepare a conda env

```bash
conda create -n mvl-challenge python=3.9
```
2. Clone the `mvl-tookit`:

```bash
git clone https://github.com/mvlchallenge/mvl_toolkit.git
cd mvl_toolkit
git submodule update --init --recursive
```

Note that we have included [`HorizonNet`](https://github.com/sunset1995/HorizonNet) as a submodule. This submodule is intended solely as an out-of-the-box layout estimation example. You will find a non-empty folder named `HorizonNet` located within the `mvl_challenge/models` directory.


3. Install `mvl-toolkit`
```bash
pip install -e .
```

4. Test installation. 
```bash
python test_toolkit.py
```

## Datasets

For this challenge, two multi-view datsets are used, [MP3D-FPE](https://github.com/EnriqueSolarte/direct_360_FPE), and [HM3D-MVL](https://github.com/mvlchallenge/mvl_toolkit/edit/mvl_chellenge_dev), both collected in equirectangular camera projection. To make them easier to work with, we have organized both datasets into a standardized and more user-friendly format. The naming convention for each frame is `${scene_name}_${version}_${room}_${idx}`, e.g., `E9uDoFAP3SH_1_room0_982`. 

The whole dataset for this challenge is presented in the following structure:
```
└── ${MVL_DATA_DIR}/
    ├── geometry_info/
    │    └── pa4otMbVnkk_0_room0_109.json
    │    └── k1cupFYWXJ6_1_room10_98.json
    │    └── ...
    ├── img/
    │    └── pa4otMbVnkk_0_room0_109.jpg
    │    └── k1cupFYWXJ6_1_room10_98.jpg
    │    └── ...
    └── labels/
          └── gt/
          │    └── pa4otMbVnkk_0_room0_109.jpg
          │    └── k1cupFYWXJ6_1_room10_98.jpg
          │    └── ...
          └── gt_vis/
               └── pa4otMbVnkk_0_room0_109.jpg
               └── k1cupFYWXJ6_1_room10_98.jpg
               └── ...
```
In relation to this dataset, there exists a group of JSON files, refered as scene list files, which contain an organized sequence of frames. These files define the training, testing, and pilot splits for each phase challenge. 
It is crucial to keep in mind that no ground truth labels are disclosed for this challeneg, except for the pilot split.

All of the scene list files predefined for this challenge can be found at `mvl_challenge/data/scene_list`

## MVL-Toolkit usage.

### Download the dataset

To downlaod the dataset for this challenege please 
```sh
python download_mvl_data.py
```

### Cheking data using scene list files

```sh
python check_scene_list.py
```

### Loading a set of data as `List <Layou class>`

`mvl_challenge/data/mp3d_fpe/{split}__{type}__scene_list.json` represent the `scene_list` which lists the data in `MVL_DATA_FORMAT`. For example, `test__gt_labels__scene_list.json` lists all the data of ground truth labels we have in `MVL_DATA_FORMAT`.

**\*Important\***: the `scene_list` will be the key when we want to access different types of the existing data.

Now we can load the data that we just downloaded and visualize them.  
For exmaple:
```bash
# You can change the scene list json file to load different data
python mvl_challenge/mvl_data/load_mvl_dataset.py -d {MVL_DATA_DIR} -f mvl_challenge/data/mp3d_fpe/test__gt_labels__scene_list.json
```

See `python mvl_challenge/mvl_data/load_mvl_dataset.py -h` for more detail.

While running the program, you should see a sequence of panorama images (i.e., the dataset images) showing up on the screen.

![Alt text](markdown/toolkit_load_data.gif)

### Example how to estimate layout within a `List <Layou class>`

Now we can load the data and predict the layout by the pre-trained model of HorizonNet.
```bash
python mvl_challenge/mvl_data/load_and_eval_mvl_dataset.py -d {MVL_DATA_DIR} -f mvl_challenge/data/mp3d_fpe/test__mp3d_fpe__scene_list.json
```

See `mvl_challenge/mvl_data/load_and_eval_mvl_dataset.py -h` for more detail.

While running the program, you should see a sequence of panorama images with the layout prediction predicted by HorizonNet showing up on the screen.

![Alt text](markdown/toolkit_evaluate.gif)

In the end, it will pop out a window showing the point cloud of all the layout estimation in 3D.

![Alt text](markdown/toolkit_point_cloud.gif)

### Save estimations

In this part, we can further save the prediction result into npz files. Each image frame will have a correspinding npz file containing the layout estimation output. These npz files will be stored in `{RESULTS_DIR}`.

```bash
python mvl_challenge/challenge_results/create_npz_files.py -d {MVL_DATA_DIR} -f mvl_challenge/data/mp3d_fpe/pilot_split__mp3d_fpe__scene_list.json -o {RESULTS_DIR}
```

See `python mvl_challenge/challenge_results/create_npz_files.py -h` for more detail.

Next, we are going to zip all the prediction npz files.

```bash
python mvl_challenge/challenge_results/create_zip_results.py -d {RESULTS_DIR} -f mvl_challenge/data/mp3d_fpe/pilot_split__mp3d_fpe__scene_list.json
```

The resulting zip file is the only file that will be submmitted to the EvalAI server.


### Submit estimations to [Eval Ai](https://eval.ai/web/challenges/challenge-page/1906/)

We recommend you to submmit the file using CLI.

```bash
# Install evalai-cli
pip install evalai
```
```bash
# Add your EvalAI account token to evalai-cli
# You can get {YOUR_TOKEN} on EvalAI
evalai set_token {YOUR_TOKEN}
```
```bash
# Submit the file
# Take warmup phase for example
evalai challenge 1906 phase 3801 submit --file {RESULTS_ZIP_FILE} --large
```

You can find more detail (e.g., changing the phase) on EvalAI website.

### Check evaluation result

We provide a `pilot split` to let you double check if the evaluation result in your local computer agrees with the one on EvalAI.

```bash
python mvl_challenge/challenge_results/evaluate_results.py -d {MVL_DATA_DIR} -f mvl_challenge/data/mp3d_fpe/pilot_split__mp3d_fpe__scene_list.json -o {PILOT_EVAL_DIR}
```

P.S. this can be done because we release 1% of the ground truth labels in `pilot split`.
