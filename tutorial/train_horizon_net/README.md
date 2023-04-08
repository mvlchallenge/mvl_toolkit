## Tutorial 1: Train your HorizonNet on our dataset

In this tutorial, we will walk you through how to train your own model using scenes in the pilot split (with labels). Make sure you have downloaded HorizonNet following [here](https://github.com/mvlchallenge/mvl_toolkit/tree/main#installation).

### Step 1: download pre-trained models
```bash
# use -h for more details
python download_pretrained_ckpt.py
# or specify download info and output directory
python download_pretrained_ckpt.py -f $DOWNLOAD_INFO -o $OUTPUT_DIR
```
Afterwards, you should see all pre-trained models under `mvl_challenge/assets/ckpt`.

### Step 2: start training HorizonNet

Instead of using the native [HorionNet](https://github.com/sunset1995/HorizonNet), we provide a wrapper at `mvl_challenge/models/wrapper_horizon_net.py` allowing you to easily train your model on the provided dataset.

This wrapper keeps the overall structure of the original implementation, except for taking out the post-processing part, which allows estimations of non-Manhattan scenes. You can find all controlable settings and hyperparameters in `train_hn.yaml`, note that parameters marked with `<Required>` should be filled.

```bash
python train_hn.py
# or
python train_hn.py --cfg "train_hn.yaml" --pilot_scene_list $SCENE_LIST -ckpt $CHECK_POINT
```


At the same time, you can find the best-performing weights on 2D and 3D IoU and evalution logging in `mvl_challenge/assets/data/mvl_training_results/hn_mp3d__scene_list__warm_up_pilot_set/ckpt`. While the tensorboard event files will be in `mvl_challenge/assets/data/mvl_training_results/hn_mp3d__scene_list__warm_up_pilot_set/log`.

Run the following command to visualize the training status
```bash
LOG_DIR="mvl_challenge/assets/data/mvl_training_results/hn_mp3d__scene_list__warm_up_pilot_set/log"
tensorboard --log_dir $LOG_DIR
```

![](https://user-images.githubusercontent.com/67839539/230726425-4d5db4a9-e495-4f29-83c2-c4b1666f7732.png)
