# coding:utf-8

import cv2
import numpy as np

import socket
import sys
import os
import time
import struct

from client.preprocess import *


def calc_compression():
    compressor = CompressorObj()
    preprocessor = Preprocessor()
    imgs_path = './cocoapi/images/val2017'
    img_names = sorted(os.listdir(imgs_path))
    img_paths = [os.path.join(imgs_path, img_name) for img_name in img_names]
    quality = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    savedStdout = sys.stdout
    for q in quality:
        with open('./logs/log_ratio_end2end_jpeg_q_{}.log'.format(q), 'w') as file:
            sys.stdout = file
            print('JPEG_END2END q={}'.format(q))
            for img_name in img_paths:
                with open(img_name, 'rb') as f:
                    img_bytes = f.read()
                    len_jpgimg = len(img_bytes)
                    res_bytes = preprocessor.inference(img_bytes, q)
                    print(len_jpgimg, len(res_bytes)+20)
        sys.stdout = savedStdout
    print("All finished")

if __name__ == "__main__":
    calc_compression()