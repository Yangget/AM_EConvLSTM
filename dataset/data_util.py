# -*- coding: utf-8 -*-
"""
 @Author  : yangzh
 @Email   : 1725457378@qq.com
 @File    : data_show.py
"""
import os

import matplotlib.pyplot as plt


def get_label(y_da):
    if 1 in y_da:
        return 1
    elif 2 in y_da:
        return 2
    elif 3 in y_da:
        return 3
    elif 4 in y_da:
        return 4
    else:
        return 0


def data_show(y):
    list_set = set(y)

    index = []
    y_ = []
    for item in list_set:
        print("the %d has found %d" % (item, y.count(item)))
        index.append(item)
        y_.append(y.count(item))

    plt.xlabel("Video clip data type")
    plt.ylabel("Data quantity")
    plt.title("Video data type and data quantity")

    plt.bar(index, y_)
    for a, b in zip(index, y_):
        plt.text(a, b + 0.05, '%.0f' % b, ha = 'center', va = 'bottom', fontsize = 11)

    plt.savefig('../dataset/data_type_and_quantity.png')
    plt.show( )


def data_show_more(y_train, y_test):
    list_train = set(y_train)
    list_test = set(y_test)

    index = []
    y_tr = []
    for item in list_train:
        print("the %d has found %d" % (item, y_train.count(item)))
        index.append(item)
        y_tr.append(y_train.count(item))

    y_te = []
    for item in list_test:
        print("the %d has found %d" % (item, y_test.count(item)))
        y_te.append(y_test.count(item))


def check_file_exit(x):
    for img_flie in x:
        for i_f in img_flie:
            if os.path.exists(i_f):
                continue
            else:
                print(i_f)
