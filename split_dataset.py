
import argparse
import shutil
import random
import os

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', help='Path to data dir')
parser.add_argument('--out_dir', type=str, default='./split_data', help='Path to out dir')
parser.add_argument('--train_prop', type=int, default=0.6, help='Proportion of training samples')
parser.add_argument('--valid_prop', type=int, default=0.2, help='Proportion of validation samples')
args = parser.parse_args()


def split_data(arr):
    num = len(arr)
    train_num = int(num * args.train_prop)
    valid_num = int(num * args.valid_prop)
    random.shuffle(arr)
    return arr[: train_num], arr[train_num: train_num+valid_num], arr[train_num+valid_num:]

def copy_file(arr, org_path, out_path):
    for i in range(len(arr)):
        file_name = os.path.join(org_path, arr[i])
        out_file_name = os.path.join(out_path, arr[i])
        shutil.copyfile(file_name, out_file_name)

def begin():
    if args.train_prop + args.valid_prop > 1:
        print('train_prop + valid_prop should be (0, 1).')
        exit(1)
    data_path = args.data_dir
    outpath = args.out_dir
    if os.path.exists(outpath):
        shutil.rmtree(outpath)
    classes = []
    for c in os.listdir(data_path):
        if os.path.isdir(os.path.join(data_path, c)):
            classes.append(c)
    if len(classes) < 2:
        print("There should be at least two classes.")
        exit(1)
    for c in classes:
        d_path = os.path.join(data_path, c)
        data_arr = os.listdir(d_path)
        if len(data_arr) < 5:
            print("Too few samples for", c)
            exit(1)
        path_train = os.path.join(outpath, 'train', c)
        path_valid = os.path.join(outpath, 'valid', c)
        path_test = os.path.join(outpath, 'test', c)
        os.makedirs(path_train)
        os.makedirs(path_valid)
        os.makedirs(path_test)
        train_arr, valid_arr, test_arr = split_data(data_arr)
        copy_file(train_arr, d_path, path_train)
        copy_file(valid_arr, d_path, path_valid)
        copy_file(test_arr, d_path, path_test)
        print(c, '/ Train:', len(train_arr), '/ Valid:', len(valid_arr), '/ Test:', len(test_arr))

if __name__ == "__main__":
    begin()
