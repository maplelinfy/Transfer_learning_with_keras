# Transfer_learning_with_keras

## Usage:

### Step1: Split the dataset for training, validation and test

python split_dataset.py --data_dir=YOUR_DATA_PATH

In your data path, each category is a folder containing all of its samples, a folder named 'split_data' is generated by default, which has 'train', 'valid', 'test' in it. 

Also, you can set the output directory of the data, proportion of training and validation samples.

### Step2: Train a model

python finetune_keras.py --data_dir=DATA_PATH_FROM_STEP1

The input here is the data folder generated from the previous step, './split_data' for example. 

We currently support only three models ["inception_v3", "vgg16", "resnet50"]. We will freeze all layers and train only the top layers at first, then we will freeze the bottom N layers and train the remaining top layers.

Reference: https://keras.io/applications/
