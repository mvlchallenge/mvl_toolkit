## Overview

This toolkit is part of the Multi-view Layout Estimation Challenge of the [Omnidirectional Computer Vision (OmniCV) workshop](https://sites.google.com/view/omnicv2023/home?authuser=0) at [CVPR'23](https://cvpr2023.thecvf.com/). To participate and submit results, join us on [EvalAI](https://eval.ai/web/challenges/challenge-page/1906/).
Please visit our official site [mvl-challege](https://sites.google.com/view/omnicv2023/challenges/multi-view-layout-challenge?authuser=0) for more detailed information .

For public queries, discussion, and free access tutorials please join us in our [Slack workspace](https://join.slack.com/t/mvl-challenge/shared_invite/zt-1m95ef0hy-ViG7fSeTt1EqiosRlZoDvQ).

## Usage

This implementation offers the following capabilities to support the mvl-challenge:

- Download the training, testing and pilot datasets used in this challenge.
- Load and register the data as an instance `<Layout class>`, that simplifies camera and layout projection.
- Load and retrive multiple `<Layout class>` instances associated to a query room.
- Provide examples of how to evaluate a layout estimation using `<Layout class>` instances.
- Provide methods to save, zip and submit layout estimates to EvalAI.

## Installation

```bash
git clone https://github.com/mvlchallenge/mvl_toolkit.git
cd mvl_toolkit
git submodule update --init --recursive
pip install -e .
```
Note that we have included [`HorizonNet`](https://github.com/sunset1995/HorizonNet) as a submodule. This submodule is intended solely as an out-of-the-box layout estimation example. You should find a non-empty folder `mvl_challenge/models/HorizonNet`.

Aferwards, run the following command to test installation.
```bash
python test_toolkit.py
```

## Datasets

In this challenge, two multi-view datsets are included, 1) [MP3D-FPE](https://github.com/EnriqueSolarte/direct_360_FPE), and 2) [HM3D-MVL](https://github.com/mvlchallenge/mvl_toolkit/edit/mvl_chellenge_dev), both collected in equirectangular camera projection.

We have reorganized both datasets into a standard naming convention as `${scene_name}_${version}_${room}_${idx}`, e.g., `E9uDoFAP3SH_1_room0_982`, and the data structure is listed as follows:
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
Besides, we provide scene list files that define the training, testing, and pilot splits for each phase in this challenge at `mvl_challenge/data/scene_list`.

⚠️ No ground truths are included for this challenge, except for the pilot split.

## Setup

**Downlaod the dataset**
```sh
python download_mvl_data.py
```

**Check data with scene list files**
```sh
python check_scene_list.py
```

**Load the data and visualize them**
```bash
# You can change the scene list json files to load different data
python mvl_challenge/mvl_data/load_mvl_dataset.py -d {MVL_DATA_DIR} -f mvl_challenge/data/mp3d_fpe/test__gt_labels__scene_list.json
```

You should see the visualization of a sequence of panorama images.

![](https://user-images.githubusercontent.com/67839539/226287033-baedde2a-1775-4c94-9102-86022df0eaa1.gif)


**Estimate layouts**

The following command loads the data and estimates layouts using the pre-trained HorizonNet model.
```bash
python mvl_challenge/mvl_data/load_and_eval_mvl_dataset.py -d {MVL_DATA_DIR} -f mvl_challenge/data/mp3d_fpe/test__mp3d_fpe__scene_list.json
```

You should see the visualization of a sequence of panorama images with the green layout predicted by HorizonNet.

![](https://user-images.githubusercontent.com/67839539/226287069-1b338e93-5f39-479f-b880-59ad8ea0b916.gif)

In the end, it will pop out a window showing the point cloud of all the layout estimations in 3D.

![](https://user-images.githubusercontent.com/67839539/226287093-289e2b5c-79cc-40d9-accb-68ed97c7bb46.gif)

**Save estimations**

In order to submit to [EvalAI](https://eval.ai/web/challenges/challenge-page/1906/) and evalute, we further save the prediction result into *.npz files. Each image frame will have a correspinding *.npz file containing the layout estimation output. These *.npz files will be stored in `{RESULTS_DIR}`.

```bash
python mvl_challenge/challenge_results/create_npz_files.py -d {MVL_DATA_DIR} -f mvl_challenge/data/mp3d_fpe/pilot_split__mp3d_fpe__scene_list.json -o {RESULTS_DIR}
```

Next, we will zip all the predicted *.npz files.

```bash
python mvl_challenge/challenge_results/create_zip_results.py -d {RESULTS_DIR} -f mvl_challenge/data/mp3d_fpe/pilot_split__mp3d_fpe__scene_list.json
```

⚠️ The resulting zip file is the only file that will be submmitted to the EvalAI server.


**Submit to [EvalAI](https://eval.ai/web/challenges/challenge-page/1906/)**

We recommend you to submmit the file using CLI.

```bash
# Install evalai-cli
pip install evalai

# Add your EvalAI account token to evalai-cli
# You can get {YOUR_TOKEN} on EvalAI
evalai set_token {YOUR_TOKEN}

# Submit the file
# Take warmup phase for example
evalai challenge 1906 phase 3801 submit --file {RESULTS_ZIP_FILE} --large
```

Find out more details (e.g., changing the phase) on [EvalAI](https://eval.ai/web/challenges/challenge-page/1906/) website.

**Check evaluation result**

We provide a `pilot split` for you to double check if the evaluation result in your local computer agrees with the one on EvalAI.

```bash
python mvl_challenge/challenge_results/evaluate_results.py -d {MVL_DATA_DIR} -f mvl_challenge/data/mp3d_fpe/pilot_split__mp3d_fpe__scene_list.json -o {PILOT_EVAL_DIR}
```

P.S. this can be done because we release 1% of the ground truth labels in `pilot split`.
