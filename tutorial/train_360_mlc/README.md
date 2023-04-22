## Tutorial 2: Use 360-MLC for self-training

In this tutorial, we will show you how to leverage multi-view consistency property, to create pseudo-labels and self-train a model in the dataset without any annotation. We proposed this method, [360-MLC](https://enriquesolarte.github.io/360-mlc/), in NeurIPS 2022, and the code is available [here](https://github.com/EnriqueSolarte/360-mlc). Hope this example can give you an idea of how to tackle the problem in this challenge.

### Step 1: Download 360-MLC
```bash
# Under your target directory
git clone --recurse-submodules git@github.com:EnriqueSolarte/360-mlc.git

cd 360-mlc

# Install python dependencies
pip install -r requirements.txt

# Install MLC library
pip install .
```

### Step 2: Create MLC pseudo labels

We will use a pre-trained model, [HorionNet](https://github.com/sunset1995/HorizonNet) for example, to predict the layouts of all the images in the same room. By using the technique described in [360-MLC](https://enriquesolarte.github.io/360-mlc/), we can then create the pseudo labels of each frame.  
You can find all controlable settings and hyperparameters in `create_mlc_labels.yaml`, note that parameters marked with `<Required>` should be filled or they will be the default value.

```bash
# use -h for more details
python create_mlc_labels.py
# or
python create_mlc_labels.py -f $SCENE_LIST -o $OUTPUT_DIR -ckpt $CHECK_POINT
```

After the pseudo labels are created, you can find them in `$OUTPUT_DIR`, default will be `mvl_challenge/assets/data/mvl_data/labels/mlc__{ckpt}__scene_list__{SCENE_LIST}/`.

There will be three folders under it:  
1. mlc_label: pseudo labels (phi coordinates) in npy format
2. mlc_vis: pseudo labels visualization
3. std: standard deviation of pseudo labels

### Step 3: Self-train the model using MLC pseudo labels

By Using the pseudo labels created in [Step 2](#step-2-create-mlc-pseudo-labels), we can self-train the model in the similar way as in [Tutorial 1](https://github.com/mvlchallenge/mvl_toolkit/tree/mvl_chellenge_dev/tutorial/train_horizon_net), which the model is trained on GT labels.  
You can also find all controlable settings and hyperparameters in `train_mlc.yaml`, note that parameters marked with `<Required>` should be filled or they will be the default value.

```bash
# use -h for more details
python train_mlc.py
# or
python train_mlc.py --{split}__scene_list $SCENE_LIST_{split} -o $OUTPUT_DIR -ckpt $CHECK_POINT
```

By default, the model will be trained on the pseudo labels of the training split, and validated on the pilot split, since the pilot split has GT labels for you to do the evaluation.

At the same time, you can find the the training result in `mvl_challenge/assets/data/mvl_training_results/mlc_{ckpt}__scene_list__{split}/`

The structure will be:  
```
└── mlc_{ckpt}__scene_list__{split}/
    ├── ckpt/
    │    └── best_score.json
    │    └── cfg.yaml
    │    └── valid_evel_{epoch}.json
    └── log/
         └── tensorboard event file

```

`cfg.yaml` summarizes all the parameters that are used in the whole training process.

`valid_evel_{epoch}.json` shows the model evaluations on the pilot split, epoch 0 means the original pre-trained model, and the self-training will start from epoch 1.

`log directory` can let you visualize the training status using tensorboard.