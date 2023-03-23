## Overview

This toolkit is part of the Multi-view Layout Estimation Challenge of the [Omnidirectional Computer Vision (OmniCV) workshop](https://sites.google.com/view/omnicv2023/home?authuser=0) at [CVPR'23](https://cvpr2023.thecvf.com/). To participate and submit results, join us on [EvalAI](https://eval.ai/web/challenges/challenge-page/1906/).
Please visit our official site [mvl-challege](https://sites.google.com/view/omnicv2023/challenges/multi-view-layout-challenge?authuser=0) for more detailed information .

For public queries, discussion, and free access tutorials please join us in our [Slack workspace](https://join.slack.com/t/mvl-challenge/shared_invite/zt-1m95ef0hy-ViG7fSeTt1EqiosRlZoDvQ).

## What can you do with this toolkit?

This implementation offers the following capabilities to support the mvl-challenge:

- Download the training, testing and pilot datasets used in this challenge.
- Load and register the data as an instance of `<Layout class>`, which simplifies camera and layout projection for you.
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

Aferwards, run the following command to test the installation.
```bash
python test_mvl_toolkit.py
```

## Datasets

In this challenge, two multi-view datasets are included, 1) [MP3D-FPE](https://github.com/EnriqueSolarte/direct_360_FPE), and 2) [HM3D-MVL](https://github.com/mvlchallenge/mvl_toolkit/edit/mvl_chellenge_dev), both collected in equirectangular camera projection.

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
          │    └── pa4otMbVnkk_0_room0_109.npz
          │    └── k1cupFYWXJ6_1_room10_98.npz
          │    └── ...
          └── gt_vis/
               └── pa4otMbVnkk_0_room0_109.jpg
               └── k1cupFYWXJ6_1_room10_98.jpg
               └── ...
```

Besides, we provide scene list files that define the training, testing, and pilot splits for each phase in this challenge at `mvl_challenge/data/scene_list/`.

⚠️ No ground truths are included for this challenge, except for the pilot split.

## Tutorial

### Downlaod the dataset
```bash
# use -h for more details
python download_mvl_data.py
# or specify output directory and split
python download_mvl_data.py -o ${MVL_DATA_DIR} -split ${SPLIT}
```

### Check data with scene list

The `*.json` files under `mvl_challenge/data/scene_list/` are the scene lists. Each scene list will be the key to let you access different types of the existing data.
For example, with `scene_list__warm_up_pilot_set.json`, we can access the data of pilot set in the warm-up phase.

Run the following command to make sure the data has been correctly downloaded and whether you can access it:

```bash
# use -h for more details
python check_scene_list.py
# or
python check_scene_list.py -d ${MVL_DATA_DIR} -f ${SCENE_LIST}
```

### Load the data and visualize

```bash
# use -h for more details
python mvl_challenge/mvl_data/load_mvl_dataset.py
# or
python mvl_challenge/mvl_data/load_mvl_dataset.py -d ${MVL_DATA_DIR} -f ${SCENE_LIST}
```

You should see a sequence of panorama images, which are specified in the passed scene list filenae.

![](https://user-images.githubusercontent.com/67839539/226287033-baedde2a-1775-4c94-9102-86022df0eaa1.gif)

### Estimate the layouts and visualize

Load the data and, moreover, estimate layouts using a pre-trained model. In the following command, we use HorizonNet as a layout estimator only for didactic purposes.

```bash
# use -h for more details
python mvl_challenge/mvl_data/load_and_eval_mvl_dataset.py
# or
python mvl_challenge/mvl_data/load_and_eval_mvl_dataset.py -d ${MVL_DATA_DIR} -f ${SCENE_LIST} --ckpt ${CHECK_POINT}
```
P.S. For the `${CHECK_POINT}`, we provide an example under `mvl_challenge/assets/ckpt/`. You can use your own model and pretrained weights for sure.

You should see the visualization of a sequence of panorama images with the green layout predicted by HorizonNet.

![](https://user-images.githubusercontent.com/67839539/226287069-1b338e93-5f39-479f-b880-59ad8ea0b916.gif)

In the end, it will pop out a window showing the point cloud of all the layout estimations in 3D.

![](https://user-images.githubusercontent.com/67839539/226287093-289e2b5c-79cc-40d9-accb-68ed97c7bb46.gif)

### Save estimations

In order to submit to [EvalAI](https://eval.ai/web/challenges/challenge-page/1906/) and evalute your estimations, we further save the prediction results into `*.npz` files. Each image from the testing split will have a correspinding `*.npz` file containing the layout estimation output. These `*.npz` files will be stored in `${RESULT_DIR}`.

```bash
# use -h for more details
python mvl_challenge/challenge_results/create_npz_files.py
# or
python mvl_challenge/challenge_results/create_npz_files.py -d ${MVL_DATA_DIR} -f ${SCENE_LIST} -o ${RESULT_DIR} --ckpt ${CHECK_POINT}
```

Next, we will zip all the predicted `*.npz` files.

```bash
# use -h for more details
python mvl_challenge/challenge_results/create_zip_results.py
# or
python mvl_challenge/challenge_results/create_zip_results.py -d ${YOUR_RESULT} -f ${SCENE_LIST}
```

⚠️ This zip file is the only file that will be submitted to the EvalAI server. It is stored in `mvl_challenge/assets/npz/` by default.

### Submit to EvalAI

We recommend you to submit your results' file `${YOUR_RESULT_ZIP}` using CLI. You should [participate](https://eval.ai/web/challenges/challenge-page/1906/participate) in adavance to submit.

```bash
# Install evalai-cli
pip install evalai

# Add your EvalAI account token to evalai-cli
evalai set_token ${YOUR_TOKEN}

# Submit the your ${YOUR_RESULT_ZIP} store at mvl_challenge/assets/npz/
evalai challenge 1906 phase 3801 submit --file ${YOUR_RESULT_ZIP} --large
```

### Check evaluation result

We provide `pilot split` for you to double check if the evaluation result in your local computer agrees with the one on EvalAI.

Evaluate on your local computer:
```bash
# use -h for more details
python mvl_challenge/challenge_results/evaluate_results.py
# or
python mvl_challenge/challenge_results/evaluate_results.py -d ${MVL_DATA_DIR} -f ${PILOT_SCENE_LIST} -o ${PILOT_EVAL_DIR} --ckpt ${CHECK_POINT}
```

If the evaluation results are matching, congratulations! You've already completed the submission!
