# Transfer_learning_with_keras
Usage:

Step1: Split the dataset for training, validation and test

python split_dataset.py --data_dir=YOUR_DATA_PATH

In your data path, each category is a folder containing all of its samples, a folder named 'split_data' is generated by default, which has 'train', 'valid', 'test' in it.

Step2: Training a model

python finetune_keras.py --data_dir=YOUR_DATA_PATH2

The input here is the data folder generated from the previous step, './split_data' for example.
