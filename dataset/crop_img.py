# -*- coding: utf-8 -*-
"""
 @Author  : yangzh
 @Email   : 1725457378@qq.com
 @File    : crop_img.py
"""
"""
This code is used to cut the source data
"""

import os

import geoio


def crop(i, j, win, img, file_path):
    cropped = img.get_data(window = [i, j, win, win], return_location = True)
    dtype = geoio.constants.DICT_NP_TO_GDAL[cropped[0].dtype]
    geoio.base.create_geo_image(file_path,
                                cropped[0],
                                'GTiff',
                                cropped[1]['geo_transform'],
                                img.meta.projection_string,
                                dtype)


def main():
    file_path = "../dataset/dataset/Volcano/lower-puna-volcano/"
    new_path = "../code/dataset/train/Volcano/lower-puna-volcano/"
    win = 1000

    img_file = os.listdir(file_path)
    for img_f in img_file:
        img_path = file_path + img_f

        img = geoio.GeoImage(img_path)
        file_imformation = img.meta

        c, h, w = file_imformation['shape']
        print(c, h, w)
        print(win)
        print(int(h / win) * int(w / win))

        for i in range(int(h / win)):

            new_file = new_path + img_f.split('.')[0] + "_" + str(i)
            if not os.path.exists(new_file):
                os.makedirs(new_file)

            for j in range(int(w / win)):
                file = new_file + '/' + new_file.split('/')[-1] + '_' + str(i) + '_' + str(j) + '.tif'
                crop(i * win, j * win, win = win, img = img, file_path = file)


if __name__ == '__main__':
    main( )
