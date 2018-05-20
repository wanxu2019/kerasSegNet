# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os
import glob
from keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator
from scipy.misc import imresize
from keras.utils.np_utils import to_categorical
from functools import partial


def label_map(img, dims, n_labels):
    array_mask = np.zeros(dims, dtype="uint8")
    m = img == 255
    array_mask[m] = 1
    size = array_mask.size
    array_mask = np.asarray(list(map(partial(to_categorical, num_classes=n_labels), array_mask.reshape(-1))))
    array_mask=array_mask.reshape((size, n_labels))
    return array_mask


def generator(img_dir, mask_dir, name_list, batch_size, dims=[224, 224], n_labels=2):
    while True:
        ix = np.random.choice(np.arange(len(name_list)), batch_size)
        imgs = []
        labels = []
        for i in ix:
            # images
            # original_img = cv2.imread(os.path.join(img_dir, name_list[i]))[:, :, ::-1]
            original_img = cv2.imread(os.path.join(img_dir, name_list[i]))
            resized_img = cv2.resize(original_img,(dims[1],dims[0]))
            array_img = img_to_array(resized_img) / 255
            imgs.append(array_img)
            # masks
            original_mask = cv2.imread(os.path.join(mask_dir, name_list[i]))
            resized_mask = cv2.resize(original_mask,(dims[1],dims[0]))
            array_mask = label_map(resized_mask[:, :, 0], dims, n_labels)
            labels.append(array_mask)
        imgs = np.array(imgs)
        labels = np.array(labels)
        yield imgs, labels


def train_val_generator(batch_size):
    train_image_dir = "data/640_480/train/deform/train"
    train_label_dir = "data/640_480/train/deform/label"
    name_list = glob.glob(train_image_dir + "/*.png")
    name_list = np.asarray(name_list)
    train_index = np.random.choice(len(name_list), int(len(name_list) * 0.8), replace=False)
    val_index = np.array(list(set(np.arange(len(name_list))) - set(train_index)))
    train_name_list = list(map(lambda x: x[x.rindex("\\") + 1:], name_list[train_index]))
    val_name_list = list(map(lambda x: x[x.rindex("\\") + 1:], name_list[val_index]))
    return generator(train_image_dir, train_label_dir, train_name_list, batch_size), generator(train_image_dir,
                                                                                               train_label_dir,
                                                                                               val_name_list,
                                                                                               batch_size)
