import numpy as np
import cv2
import sys
import copy
import math
import os

import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append('./')
from utils.imgutils import *

def plot_filters(weights):
    def prime_powers(n):
        """
        Compute the factors of a positive integer
        Algorithm from https://rosettacode.org/wiki/Factors_of_an_integer#Python
        :param n: int
        :return: set
        """
        factors = set()
        for x in range(1, int(math.sqrt(n)) + 1):
            if n % x == 0:
                factors.add(int(x))
                factors.add(int(n // x))
        return sorted(factors)

    def get_grid_dim(x):
        """
        Transforms x into product of two integers
        :param x: int
        :return: two ints
        """
        factors = prime_powers(x)
        if len(factors) % 2 == 0:
            i = int(len(factors) / 2)
            return factors[i], factors[i - 1]
        i = len(factors) // 2
        return factors[i], factors[i]
    weights_data = np.transpose(weights, (3, 2, 0, 1))
    out_channels, in_channels, h, w = weights_data.shape
    res = np.zeros(shape=(out_channels, h, w))
    # calculate average value of filters
    for idx, filt in enumerate(weights_data):
        dst = np.zeros(shape=(h, w))
        for cur in filt:
            dst += cur
        dst /= in_channels
        res[idx] = dst
    w_min = np.min(res)
    w_max = np.max(res)
    grid_r, grid_c = get_grid_dim(out_channels)
    fig, axes = plt.subplots(min([grid_r, grid_c]),
                             max([grid_r, grid_c]))

    for l, ax in enumerate(axes.flat):
        img = res[l]
        ax.imshow(img, vmin=w_min, vmax=w_max,
                  interpolation='nearest', cmap='seismic')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

def sort_fmaps(feature_maps, method):
    """
    @feature_maps: original float feature maps of shape (None, 78, 78, 128)
    @method:
        lum: Sort feature maps by luminance
        cluster: clustering the filters and sort the feature maps
        delta: sort the feature maps by difference to base feature map(average)
    
    return: dl: sorted list
            fmaps: feature maps
    """
    fmaps = copy.copy(feature_maps)[0]
    fmaps = (fmaps - fmaps.min()) * 255 / (fmaps.max() - fmaps.min())
    fmaps = fmaps.astype(np.uint8).transpose((2, 0, 1))
    d = {}
    dl = []
    if method == 'lum':
        for idx, fmap in enumerate(fmaps):
            d[idx] = np.average(fmap)
        dl = sorted(d.items(), key=lambda x: x[1], reverse=False)
    if method == 'cluster':
        pass
    if method == 'random':
        for idx in range(128):
            d[idx] = idx
        dl = d.items()
        np.random.shuffle(dl)
    if method == 'delta':
        ave = np.average(fmaps, axis=0)
        min_fmap = np.average(fmaps[0])
        min_fmap_idx = 0
        for idx, fmap in enumerate(fmaps):
            tmp = np.average(fmaps)
            if tmp < min_fmap:
                min_fmap = tmp
                min_fmap_idx = idx
            d[idx] = delta
        dl = sorted(d.items(), key=lambda x: x[1], reverse=False)
    return dl, fmaps


def test_h264():
    feature_maps = np.load('./dev/fmap.npy')
    dl1, feature_maps1 = sort_fmaps(feature_maps, 'delta')
    dl2, feature_maps2 = sort_fmaps(feature_maps, 'lum')
    for idx, item in enumerate(dl):
        print(idx)
        cv2.imwrite('fmaps/fmap_{:03d}.jpg'.format(idx), feature_maps[item[0]])
    return dl


def merge_imgs(path):
    shape = (8, 16)
    rows, cols = shape[0], shape[1]
    img_names = sorted(os.listdir(path))
    img_paths = [os.path.join(path, img_name) for img_name in img_names]
    cnt = 0
    tmp = []
    res = []
    for img_name in img_paths:
        tmp.append(cv2.imread(img_name, 0))
        cnt += 1
        if cnt == cols:
            cnt = 0
            res.append(np.hstack(tmp))
            tmp = []
    res_img = np.vstack(res)
    return res_img


def compare_h264():
    imgs_orig_path = './fmaps/fmaps_orig'
    # imgs_dec_path = './fmaps/fmaps_dec'
    fmap_orig = merge_imgs(imgs_orig_path)
    # fmap_dec = merge_imgs(imgs_dec_path)
    cv2.imshow('orig', fmap_orig)
    # cv2.imshow('dec', fmap_dec)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def batched_h264_test(crf, batch_size):
    basedir = './channel_images/'
    filename = 'fmap_scene_{}_frame_%d_channel_{}.jpg'
    logfile_name = './logs_pedes/log_batched_h264_crf_{}_batchsize_{}.log'
    import ffmpeg
    import random
    scene_num = 5
    frame_num = 200
    channel_num = 128
    r = frame_num // batch_size
    starts = [i*batch_size for i in range(r)]
    savedStdout = sys.stdout
    with open(logfile_name.format(crf, batch_size), 'w') as file:
        sys.stdout = file
        for s in range(scene_num):
            for c in range(channel_num):
                start = random.choice(starts)
                out, _ = (ffmpeg.input(basedir+filename.format(s, c), framerate=20, s='78x78', f='image2'). \
                    output('pipe:', preset='medium',
                        pix_fmt='yuv420p', start_number=start, vframes=batch_size, vcodec='libx264', f='h264', crf=crf). \
                    run(capture_stdout=True, quiet=True))
                print('scene:{} channel:{} start_num:{} {} {}'.format(s, c, start, batch_size, len(out)))
    sys.stdout = savedStdout

if __name__ == "__main__":
    # filters_data = np.load('./dev/filters.npy')
    # plot_filters(filters_data)
    # fmap_data = np.load('./dev/fmap.npy')
    # feature_maps_to_image(fmap_data, is_display=True)
    # test_h264()
    # compare_h264()
    crfs = [20, 22, 24]    
    bs = [2, 4, 6, 8, 10]
    for crf in crfs:
        for b in bs:
            batched_h264_test(crf, b)