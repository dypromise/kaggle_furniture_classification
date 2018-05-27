"""this py file contains metheds to make data list from data dir
"""

import os
import shutil
import csv
from imp import reload
import pandas as pd
from PIL import Image

data_root = '/mnt/lustre17/yangkunlin/fur_dy/data'

train_dir = os.path.join(data_root, 'train_ori')
val_dir = os.path.join(data_root, 'valid')
test_dir = os.path.join(data_root, 'test')

train_csv = os.path.join(data_root, 'train.csv')
val_csv = os.path.join(data_root, 'val.csv')
test_csv = os.path.join(data_root, 'test.csv')

val_list = os.path.join(data_root, 'val_list.txt')


def func1():
    for i in range(128):
        tmp_dir_val = os.path.join(val_dir, '%d' % (i+1))
        dest_dir = os.path.join(train_dir, '%d' % (i+1))
        for item in os.listdir(tmp_dir_val):
            src_path = os.path.join(tmp_dir_val, item)
            des_path = os.path.join(dest_dir, item)
            if(os.path.isfile(src_path)):
                shutil.copyfile(src_path, des_path)


def func2():
    with open(train_csv, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['img', 'label'])
        for i in range(128):
            train_sub_dir = os.path.join(train_dir, '%d' % (i+1))
            for item in os.listdir(train_sub_dir):
                # labels begin from 0! matching Pytorch!
                data = ['%d' % (i+1)+'/'+item, '%d' % (i)]
                writer.writerow(data)

    with open(val_csv, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['img', 'label'])
        for i in range(128):
            train_sub_dir = os.path.join(val_dir, '%d' % (i+1))
            for item in os.listdir(train_sub_dir):
                data = ['%d' % (i+1)+'/'+item, '%d' % (i)]
                writer.writerow(data)

    # with open(val_csv, 'w') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['img', 'label'])

    #     val_list_f = open(val_list, 'r')
    #     lines = val_list_f.readlines()
    #     for line in lines:
    #         image_name = line.split(' ')[0]
    #         label = int(line.split(' ')[1])
    #         data = [image_name, '%d' % (label)]
    #         writer.writerow(data)

    with open(test_csv, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['img', 'label'])
        for item in os.listdir(test_dir):
            img_idx = int(item.split('.')[0])
            data = [item, '%d' % (img_idx)]
            writer.writerow(data)


def print_size():
    for img in os.listdir(os.path.join(data_root, 'val')):
        img_name = os.path.join(data_root, 'val', img)
        image = Image.open(img_name).convert('RGB')
        print(image.size)
