# -*- coding: utf-8 -*-
"""
 @Author  : yangzh
 @Email   : 1725457378@qq.com
 @File    : data_detail.py
"""
import os

import cv2
import numpy as np
import pandas as pd
from keras.utils import Sequence
from keras.utils.np_utils import to_categorical

from dataset.data_util import get_label

class BaseSequence(Sequence):

    def __init__(self, image_filenames, labels, classes, batch_size, input_size, Len):
        self.image_filenames, self.labels = image_filenames, labels
        self.classes = classes
        self.batch_size = batch_size
        self.input_size = input_size
        self.Len = Len

    def __len__(self):
        return int(len(self.image_filenames) / self.batch_size)

    def preprocess_img(self, img_path):
        img = cv2.imread(img_path[1:])
        img = cv2.resize(img, (self.input_size, self.input_size), interpolation = cv2.INTER_AREA)
        img = np.array(img)
        img = img[:, :, ::-1]
        return img

    def __getitem__(self, idx):
        x = []
        y = []

        self.batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size][0]
        self.batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size][0]

        self.batch_x = [self.preprocess_img(img_path) for img_path in self.batch_x]
        x.append(self.batch_x)
        y.append(self.batch_y)

        y = to_categorical(y, num_classes = 5)
        x = np.array(x)
        y = np.array(y[:])
        return x, y


def get_all_csv(path):
    all_csv = []

    for root, dirs, files in os.walk(path):
        if files == []:
            continue
        elif files[0][-1] == 'v':
            for cvs_file in files:
                cvs_file = root + "/" + cvs_file
                all_csv.append(cvs_file)
    return all_csv


def getdata(path, batch_size, input_size, Len, classes):
    # get csv data
    all_csv = get_all_csv(path)

    import random
    random.shuffle(all_csv)

    print("the dataset count %d" % (len(all_csv)))
    x = []
    y = []

    # get index
    for t_f in all_csv:
        t_d = pd.read_csv(t_f)
        x.append(list(t_d['FileName']))
        y.append(list(t_d['type']))

    data_x = []
    data_y = []

    # data slicing
    for x_f, y_f in zip(x, y):
        for i in range(int(len(x_f) / Len)):
            x_da = x_f[i * Len:i * Len + Len]
            y_da = y_f[i * Len:i * Len + Len]
            if get_label(y_da) == 0:
                continue
            else:
                data_x.append(x_da)
                data_y.append(get_label(y_da))

    data_x = random.sample(data_x,36)
    data_y = random.sample(data_y,36)
    print(data_x)

    # from sklearn.utils import class_weight
    # class_weight = class_weight.compute_class_weight('balanced',np.unique(data_y),data_y)

    # Visualization data
    # data_show(data_y)

    # split training set and testing machine
    #     x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.1, shuffle = True)

    # Visualization more data
    # data_show_more(y_train, y_test)

    train_sequence = BaseSequence(data_x, data_y, classes, batch_size, input_size, Len)

    return train_sequence


if __name__ == '__main__':
    path = "./dataset/train_csv/"
    train_sequence = getdata(path, batch_size = 1, input_size = 50, Len = 5, classes = 3)
    for i in range(100):
        batch_data, bacth_label = train_sequence.__getitem__(i)
        # print(batch_data.shape)
        # print(bacth_label)
        #
