## Tutorial 1

In this tutorial, we will walk you through how to train your own model using pilot scenes. Make sure you have downloaded HorizonNet following [here](https://github.com/mvlchallenge/mvl_toolkit/blob/mvl_chellenge_dev/readme.md#installation).

### Step 1: download pre-trained models
```bash
python download_pretrained_ckpt.py
# or specify download info and output directory
python download_pretrained_ckpt.py -f {DOWNLOAD_INFO} -o {OUTPUT_DIR}
```
Afterwards, you should see all pre-trained models under `mvl_challenge/assets/ckpt`.

### Step 2: start training HorizonNet
```bash
python train_hn.py
# or
python train_hn.py --pilot_scene_list {PILOT_SCENE_LIST} -ckpt {CKPT_PATH} --cuda_device {GPU_INDEX}
```
At the same time, you can find the best-performing weights on 2D and 3D IoU and evalution logging in `mvl_challenge/assets/data/mvl_training_results/hn_mp3d__scene_list__warm_up_pilot_set/ckpt`. While the tensorboard event files will be in `mvl_challenge/assets/data/mvl_training_results/hn_mp3d__scene_list__warm_up_pilot_set/log`.

Run the following command to visualize the training status
```bash
LOG_DIR="mvl_challenge/assets/data/mvl_training_results/hn_mp3d__scene_list__warm_up_pilot_set/log"
tensorboard --log_dir $LOG_DIR
```