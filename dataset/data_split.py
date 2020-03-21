# -*- coding: utf-8 -*-
"""
 @Author  : yangzh
 @Email   : 1725457378@qq.com
 @File    : data_split.py
"""
"""
This code is used to split the dataset
"""
import os
import shutil


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


def move(path, csv):
    for c_i in csv:
        new_p = path + c_i.split('/')[3] + '/' + c_i.split('/')[4] + '/'
        print(new_p)
        if not os.path.exists(new_p):
            os.makedirs(new_p)
        shutil.copy(c_i, new_p)


def split(train_path, val_path, all_csv):
    import random
    random.shuffle(all_csv)

    CSV_len = len(all_csv)
    train, val = all_csv[:int(CSV_len * 0.5)], all_csv[int(CSV_len * 0.5):]

    move(train_path, train)
    move(val_path, val)
    print(len(train), len(val))


if __name__ == '__main__':
    path = "../dataset/train_csv/"
    train_path = "./csv/train_csv/"
    test_path = "./csv/test_csv/"
    Len = 5

    # get csv data
    all_csv = get_all_csv(path)

    split(train_path, test_path, all_csv)
