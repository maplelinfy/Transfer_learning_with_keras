
'''
Model	    Size	Parameters	 Depth
VGG16	    528 MB	138,357,544	 23
ResNet50	99 MB	25,636,712	 168
InceptionV3	92 MB	23,851,784	 159
'''

from keras.applications.imagenet_utils import preprocess_input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense, Dropout
from keras.models import load_model
from keras.models import Model
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras import regularizers
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
import numpy as np
import argparse
import glob
import math
import os


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', help='Path to data dir')
parser.add_argument('--model', type=str, default="inception_v3", help='Your pre-trained classification model of choice',
                    choices=['vgg16', 'inception_v3', 'resnet50'])
parser.add_argument('--mode', type=int, default=0, help='Train:0; Test:1; Predict(unknow_results):2', choices=[0, 1, 2])
parser.add_argument('--batch_size', type=int, default=32, help='Number of images in each batch')
parser.add_argument('--freeze_layers_number', type=int, default=1, help='Number of training layers in model')
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout ratio')
parser.add_argument('--continue_training', type=bool, default=False, help='Whether to continue training from a checkpoint')
args = parser.parse_args()


def get_file_num(dir):
    num = 0
    subdir = os.listdir(dir)
    for s in subdir:
        path = os.path.join(dir, s)
        num += len(os.listdir(path))
    return num

def get_classes_from_data_dir(dir):
    return sorted([o for o in os.listdir(dir) if os.path.isdir(os.path.join(dir, o))])

def preprocess_input_tf(x):
    return preprocess_input(x, mode='tf')

def load_img(img_path):
    img = image.load_img(img_path, target_size=(img_h, img_w))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return preprocess_input_tf(x)[0]

def log_loss(y_true, y_pred):
    summ = 0.0
    for i in range(len(y_true)):
        summ -= math.log(max(y_pred[i][y_true[i]], 1e-10))
    return summ / len(y_true)

def get_datagen(data_dir, *arg, **kwargs):
    idg = ImageDataGenerator(*arg, **kwargs)
    return idg.flow_from_directory(data_dir, target_size=(img_h, img_w), classes=classes)

def get_callbacks(weights_path, patience=10, monitor='val_loss'):
    early_stopping = EarlyStopping(verbose=1, patience=patience, monitor=monitor)
    model_checkpoint = ModelCheckpoint(weights_path, save_best_only=True, save_weights_only=True, monitor=monitor)
    return [early_stopping, model_checkpoint]

def get_class_weight(d):
    white_list_formats = {'png', 'jpg', 'jpeg', 'bmp'}
    class_number = dict()
    dirs = sorted([o for o in os.listdir(d) if os.path.isdir(os.path.join(d, o))])
    k = 0
    for class_name in dirs:
        class_number[k] = 0
        iglob_iter = glob.iglob(os.path.join(d, class_name, '*.*'))
        for i in iglob_iter:
            _, ext = os.path.splitext(i)
            if ext[1:] in white_list_formats:
                class_number[k] += 1
        k += 1
    total = np.sum(list(class_number.values()))
    max_samples = np.max(list(class_number.values()))
    mu = 1. / (total / float(max_samples))
    keys = class_number.keys()
    class_weight = dict()
    for key in keys:
        score = math.log(mu * total / float(class_number[key]))
        class_weight[key] = score if score > 1. else 1.
    return class_weight

def build_finetune_model(base_model, fc_layers_list):
    num_classes = len(classes)
    layer_num = len(base_model.layers)
    for i in range(layer_num):
        base_model.layers[i].trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    for fc in fc_layers_list:
        x = Dense(fc, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
        x = Dropout(args.dropout)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    finetune_model = Model(inputs=base_model.input, outputs=predictions)
    return finetune_model

def CNN_model():
    base_model = None
    if args.model == "vgg16":
        from keras.applications.vgg16 import VGG16
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_h, img_w, 3))
    elif args.model == "resnet50":
        from keras.applications.resnet50 import ResNet50
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_h, img_w, 3))
    elif args.model == "inception_v3":
        from keras.applications.inception_v3 import InceptionV3
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(img_h, img_w, 3))
    add_list = [1024]
    finetune_model = build_finetune_model(base_model, fc_layers_list=add_list)
    return finetune_model

def freeze_top_layers(model):
    for layer in model.layers[:args.freeze_layers_number]:
        layer.trainable = False
    for layer in model.layers[args.freeze_layers_number:]:
        layer.trainable = True

def model_compile(model, train_dir, valid_dir, opt, n_epochs):
    train_num = get_file_num(train_dir)
    valid_num = get_file_num(valid_dir)
    print(train_num, " training samples and ", valid_num, " validate samples.")
    if args.continue_training:
        model.load_weights(os.path.join(model_path, args.model + "_checkpoint"))
    model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
    train_data = get_datagen(train_dir,
                             rotation_range=30.,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             preprocessing_function=preprocess_input_tf)
    callbacks = get_callbacks(os.path.join(model_path, args.model + '_checkpoint'))
    class_weight = get_class_weight(train_dir)
    model.fit_generator(train_data,
                        steps_per_epoch=train_num / float(args.batch_size),
                        epochs=n_epochs,
                        validation_data=get_datagen(valid_dir, preprocessing_function=preprocess_input_tf),
                        validation_steps=valid_num / float(args.batch_size),
                        callbacks=callbacks,
                        class_weight=class_weight)
    return model

def predict(outfile):
    model_file = os.path.join(model_path, args.model + '_finetune.h5')
    model = load_model(model_file)
    img_path = args.data_dir
    img_arr = os.listdir(img_path)
    f = open(outfile, 'w')
    batch = []
    for i in range(len(img_arr)):
        x = load_img(os.path.join(img_path, img_arr[i]))
        batch.append(x)
        if (i + 1) % args.batch_size == 0:
            res = model.predict(np.array(batch))
            res = np.argmax(res, axis=1)
            for j in range(len(res)):
                file_name = os.path.join(img_path, img_arr[i + 1 - args.batch_size + j])
                f.write(file_name+'\t' + classes[j] + '\n')
            batch = []
        if i == len(img_arr) - 1 and len(batch) != 0:
            res = model.predict(np.array(batch))
            res = np.argmax(res, axis=1)
            for j in range(len(res)):
                file_name = os.path.join(img_path, img_arr[i + 1 - len(batch) + j])
                f.write(file_name + '\t' + classes[j] + '\n')
    f.close()

def test():
    model_file = os.path.join(model_path, args.model + '_finetune.h5')
    model = load_model(model_file)
    test_dir = os.path.join(args.data_dir, 'test')
    test_class = os.listdir(test_dir)
    print("###############################################################")
    for c in test_class:
        if c not in classes: continue
        tc = classes.index(c)
        dir = os.path.join(test_dir, c)
        test_arr = os.listdir(dir)
        y_true = len(test_arr) * [tc]
        res = []
        for i in range(len(test_arr)):
            x = load_img(os.path.join(dir, test_arr[i]))
            res0 = model.predict(np.array([x]))
            res.append(res0[0])
        res_index = np.argmax(res, axis=1)
        acc = accuracy_score(y_true, res_index)
        logloss = log_loss(y_true, res)
        print(c, "/", "Acc:", acc, "/", "LogLoss:", logloss)
    print("###############################################################")

def train():
    train_dir = os.path.join(args.data_dir, 'train')
    valid_dir = os.path.join(args.data_dir, 'valid')
    model = CNN_model()
    adam = Adam(lr=1e-4)
    model = model_compile(model, train_dir, valid_dir, adam, 20)
    freeze_top_layers(model)
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model = model_compile(model, train_dir, valid_dir, sgd, 20)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model.save(os.path.join(model_path, args.model + '_finetune.h5'))
    joblib.dump(classes, os.path.join(model_path, args.model + '_class'))


if __name__ == "__main__":
    classes = []
    model_path = './model'
    if args.mode == 0:
        dir = os.path.join(args.data_dir, 'train')
        classes = get_classes_from_data_dir(dir)
    else:
        classes = joblib.load(os.path.join(model_path, args.model + '_class'))

    img_h, img_w = 0, 0
    if args.model == "vgg16":
        img_h, img_w = 224, 224
    elif args.model == "resnet50":
        img_h, img_w = 299, 299
    elif args.model == "inception_v3":
        img_h, img_w = 299, 299
    else:
        ValueError("The model you requested is not supported in Keras")
        exit(1)
    import time
    print(time.ctime())
    if args.mode == 0:
        train()
    elif args.mode == 1:
        test()
    else:
        predict('prediction_results.xls')
    print(time.ctime())
