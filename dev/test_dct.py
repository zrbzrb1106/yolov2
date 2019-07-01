import numpy as np
import cv2
import sys
import copy
import zlib
import seaborn as sns
import matplotlib.pyplot as plt
import webp
import io
from PIL import Image
from skimage.feature import *
from sklearn.cluster import KMeans
import os

from heapq import heappush, heappop, heapify
from collections import defaultdict

sys.path.append('./')
from utils.imgutils import *

def test_webp():
    feature_maps = np.load('./dev/fmap.npy')
    fmap_img = feature_maps_to_image(feature_maps, (8, 16))[0][0]
    encode_param_jpg = [int(cv2.IMWRITE_JPEG_QUALITY), 20]
    # cv2.imwrite('./featuremaps_20.jpg', fmap_img, encode_param_jpg)
    # encode_param_png = [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
    cv2.imwrite('./featuremaps.png', fmap_img, encode_param_jpg)
    # im = Image.fromarray(fmap_img)
    # im.save('./featuremaps_20.webp', 'webp', quality=20)


def test_pil_palette():
    # img = cv2.imread('./img2.jpg')
    img = Image.open('./featuremaps.png')
    arr = np.array([7, 24, 86, 236, 255, 255, 255, 255,
                    57, 60, 112, 88, 255, 255, 255, 255,
                    146, 176, 199, 173, 255, 255, 255, 255,
                    247, 171, 199, 185, 255, 255, 255, 255,
                    255, 255, 255, 255, 255, 255, 255, 255,
                    255, 255, 255, 255, 255, 255, 255, 255,
                    255, 255, 255, 255, 255, 255, 255, 255,
                    255, 255, 255, 255, 255, 255, 255, 255])
    arr2 = np.array([16, 11, 10, 16, 24, 40, 51, 61,
                     12, 12, 14, 19, 26, 58, 60, 55,
                     14, 13, 16, 24, 40, 57, 69, 56,
                     14, 17, 22, 29, 51, 87, 80, 62,
                     18, 22, 37, 56, 68, 109, 103, 77,
                     24, 35, 55, 64, 81, 104, 113, 92,
                     49, 64, 78, 87, 103, 121, 120, 101,
                     72, 92, 95, 98, 112, 100, 103, 99])
    arr3 = np.array(
        [
            1, 1, 255, 255, 255, 255, 255, 255,
            1, 1, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255
        ]
    )
    arr = (arr).astype(np.int)
    qt = {0: arr3}
    from PIL import JpegImagePlugin
    qt = JpegImagePlugin.convert_dict_qtables(qt)
    file = io.BytesIO()
    img_qt = img.save(file, 'JPEG', qtables=qt)
    file.seek(0)
    img_data = file.read()
    img__ = cv2.imdecode(np.frombuffer(img_data, dtype=np.uint8), 1)
    cv2.imshow('tmp', img__)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    qt_n = img__.quantization
    print(0)


def calc_hist(arr):
    data = arr.ravel()
    d = defaultdict(int)
    num = len(data)
    for idx, i in enumerate(data):
        if i not in d.keys():
            d[i] = 0
        d[i] += 1
    d = sorted(d.items(), key=lambda x: x[1], reverse=True)
    return d


def calc_bin_stat(arr):
    data = arr.ravel()
    res = []
    from itertools import groupby
    for k, g in groupby(data, key=lambda x: x // 0.01):
        res.append('{}-{}: {}'.format(k*0.01, (k+1)*0.01, len(list(g))))
    return res


def space_values_prune(arr, thres=0.001):
    data = copy.copy(arr)
    data = data.astype(np.float16)
    data[np.abs(data) < thres] = 0
    return data


def space_values_digitize(arr):
    pass


def get_huffmancode(arr):
    hist_dict = calc_hist(arr)
    heap = [[wt, [sym, ""]] for sym, wt in hist_dict.items()]
    heapify(heap)
    while len(heap) > 1:
        lo = heappop(heap)
        hi = heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    res = sorted(heappop(heap)[1:], key=lambda p: (len(p[-1]), p))
    code_dict = {}
    for p in res:
        code_dict[p[0]] = p[1]
    return code_dict


def encode(arr, code_dict):
    x = copy.copy(arr)
    height, width = x.shape
    img_bit_stream_list = []
    for idx, i in enumerate(x):
        r = [code_dict[j] for j in i]
        tmp = ''.join(r)
        img_bit_stream_list.append(tmp)
    img_bit_stream = ''.join(img_bit_stream_list)
    img_bit_stream = bytes(img_bit_stream, 'utf-8')
    length = len(img_bit_stream) / 8
    print(length)
    r = zlib.compress(img_bit_stream)
    print(len(r))
    return img_bit_stream


def decode(code, code_dict):
    pass


def prune(arr, hist_dict, K):
    res = copy.copy(arr)
    tmp = sorted(hist_dict.items(), key=lambda x: x[1], reverse=True)
    d = {}
    for item in tmp[0:K]:
        d[item[0]] = item[1]
    for idxr, row in enumerate(res):
        for idxc, col in enumerate(row):
            if int(res[idxr, idxc]) not in d.keys():
                res[idxr, idxc] = 0
    return res


def calc_sparsity(arr):
    x = copy.copy(arr)
    h, w = x.shape
    total = h * w
    nonzerocnt = np.count_nonzero(x)
    return (total - nonzerocnt) / total


def get_q_matrix(q):
    Tb = np.array([16, 11, 10, 16, 24, 40, 51, 61,
                   12, 12, 14, 19, 26, 58, 60, 55,
                   14, 13, 16, 24, 40, 57, 69, 56,
                   14, 17, 22, 29, 51, 87, 80, 62,
                   18, 22, 37, 56, 68, 109, 103, 77,
                   24, 35, 55, 64, 81, 104, 113, 92,
                   49, 64, 78, 87, 103, 121, 120, 101,
                   72, 92, 95, 98, 112, 100, 103, 99])
    S = 5000 / q if q < 50 else (200 - 2*q)
    Ts = np.floor((S*Tb + 50) / 100)
    Ts[Ts == 0] = 1
    return Ts


def padding(img):

    h, w, channels = img.shape

    a = h // 8
    b = w // 8
    m = a * 8
    n = b * 8
    if m < a:
        a += 1
    if n < b:
        b += 1
    hn, wn = a*8, b*8
    res = np.zeros(shape=(hn, wn, channels))
    res[:h, :w, :] = img
    return res


def dct_compress(img, q):
    img_ = copy.copy(img)
    height, width, channels = img_.shape
    min_val, max_val = img_.min(), img_.max()
    if channels != 1:
        img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2YCrCb)
        tmp = [img_[:, :, i] for i in range(channels)]
        img_ = np.hstack(tmp)
        height, width = img_.shape
    img_ = img_ - min_val
    img_ = minmax_norm(img_, 0, 255)
    img_ = img_ - 128
    rows_num, cols_num = height // 8, width // 8
    rows = np.vsplit(img_, rows_num)
    res = np.zeros(shape=(height, width))
    res_dct = np.zeros(shape=(height, width), dtype=np.int)
    delta = 0
    for idxr, r in enumerate(rows):
        tmp = np.hsplit(r, cols_num)
        for idxc, c in enumerate(tmp):
            c_dct = cv2.dct(c)
            # b = dct_prune(c_dct, 3)
            b = dct_un_select(c_dct, q)
            b = np.round(b, 0)
            # delta = b - delta
            # btmp = copy.copy(b)
            # btmp[0][0] = 0
            # b[abs(b) < abs(btmp.max() / 50)] = 0
            # res_dct[idxr*8:idxr*8+8, idxc*8:idxc*8+8] = b
            delta = b
            bb = b
            # ib = cv2.idct(bb)
            # ib = minmax_norm(ib, c.min(), c.max())
            ib = np.round(cv2.idct(bb)) + 128
            res[idxr*8:idxr*8+8, idxc*8:idxc*8+8] = ib
    res[res < 0] = 0
    res[res > 255] = 255
    res = res.astype(np.uint8)
    res = res - res.min()
    res = minmax_norm(res, min_val, max_val)
    if channels == 3:
        tmp = np.hsplit(res, 3)
        res = np.zeros(shape=(height, width // 3, 3))
        res[:, :, 0] = tmp[0]
        res[:, :, 1] = tmp[1]
        res[:, :, 2] = tmp[2]
        res[res > 255] = 255
        res[res < 0] = 0
        res = res.astype(np.uint8)
        res = cv2.cvtColor(res, cv2.COLOR_YCrCb2BGR)
    return res, res_dct


def test_glcm():
    fmaps_path = './fmaps/fmaps_orig'
    img_names = sorted(os.listdir(fmaps_path))
    img_paths = [os.path.join(fmaps_path, img_name) for img_name in img_names]
    cnt = 0
    import time
    start = time.time()
    for img_ in img_paths:
        img = cv2.imread(img_, 0)
        print(cnt)
        cnt += 1
        tmp = copy.copy(img)
        img = (img / 32).astype(np.uint8)
        g = greycomatrix(img, [1], [0, np.pi/4, np.pi/2,
                                    3*np.pi/4], levels=8, symmetric=False, normed=True)
        contrast = greycoprops(g, 'contrast')[0][0] * 1
        energy = greycoprops(g, 'energy')[0][0] * 1
        homo = greycoprops(g, 'homogeneity')[0][0] * 1
        corr = greycoprops(g, 'correlation')[0][0] * 1
        asm = greycoprops(g, 'ASM')[0][0] * 1
        dis = greycoprops(g, 'dissimilarity')[0][0] * 1
        res = {"contrast":contrast, "energy":energy, "homogeneity":homo, "correlation":corr, "asm":asm, "dissimilarity":dis}
        print(res)
        cv2.imshow("img", tmp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    end = time.time()
    print("GLCM Analysis Time: ", end - start)


def dct_prune(macroblock, region=1):
    idx_1 = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 1)]
    idx_2 = [(0, 3), (0, 4), (0, 5), (1, 2), (1, 3), (1, 4), (2, 1),
             (2, 2), (2, 3), (3, 0), (3, 1), (3, 3), (4, 0), (4, 1), (5, 0)]
    idx_3 = [(0, 6), (0, 7), (1, 5), (1, 6), (1, 7), (2, 4), (2, 5), (2, 6), (3, 3), (3, 4), (3, 5), (4, 2), (4, 3), (4, 4), (5, 1),
             (5, 2), (5, 3), (6, 0), (6, 1), (6, 2), (7, 0), (7, 1), (7, 2)]
    idx_4 = [(2, 7), (3, 6), (3, 7), (4, 5), (4, 6), (4, 7), (5, 4),
             (5, 5), (5, 6), (6, 3), (6, 4), (6, 5), (7, 2), (7, 3), (7, 4)]
    idx_5 = [(5, 7), (6, 6), (6, 7), (7, 5), (7, 6), (7, 7)]
    idx_test = [(0, 3), (0, 4), (0, 5), (1, 3), (1, 4), (2, 3),
                (3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (5, 0)]
    idx_one = [(3, 3)]
    d = {1: idx_1, 2: idx_1 + idx_2, 3: idx_1 + idx_2 +
         idx_3, 4: idx_4, 5: idx_5, 6: idx_test, 7: idx_one}
    x = copy.copy(macroblock)
    res = np.zeros(shape=x.shape)
    idxs = d[region]
    for i in idxs:
        res[i] = x[i]
    return res


def dct_un_select(macroblock, region=1):
    r1 = [(0, 0)]
    r2 = [(0, 1), (1, 0)]
    r3 = [(0, 2), (1, 1), (2, 0)]
    r4 = [(0, 3), (1, 2), (2, 1), (3, 0)]
    r5 = [(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)]
    r6 = [(0, 5), (1, 4), (2, 3), (3, 2), (4, 1), (5, 0)]
    r7 = [(0, 6), (1, 5), (2, 4), (3, 3), (4, 2), (5, 1), (6, 0)]
    r8 = [(0, 7), (1, 6), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1), (7, 0)]
    r9 = [(1, 7), (2, 6), (3, 5), (4, 4), (5, 3), (6, 2), (7, 1)]
    r = r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8 + r9
    rr = [r1, r2, r3, r4, r5, r6, r7, r8, r9]
    x = copy.copy(macroblock)
    res = np.zeros(shape=x.shape)
    for i in r:
        res[i] = x[i]
    if region == 0:
        return res
    tmp = rr[region-1]
    for j in tmp:
        res[j] = 0.0
    return res


def plot_dct_macroblock():
    feature_maps = np.load('./dev/fmap.npy')
    fmap_image = feature_maps_to_image(feature_maps, is_digitize=1)[0][0]
    height, width = fmap_image.shape
    fmap_image = fmap_image - 128
    fmap_image = fmap_image.astype(np.float)
    rows_num, cols_num = height // 8, width // 8
    rows = np.vsplit(fmap_image, rows_num)
    res = np.zeros(shape=(8, 8, rows_num*cols_num))
    cnt = 0
    for idxr, r in enumerate(rows):
        tmp = np.hsplit(r, cols_num)
        for idxc, c in enumerate(tmp):
            c_dct = cv2.dct(c)
            res[:, :, cnt] = np.abs(c_dct)
            cnt += 1
    res_ave = np.average(res, 2)
    macroblock = res_ave.astype(np.float)
    macroblock = np.round(macroblock, 1)
    plt.imshow(macroblock, cmap=plt.cm.Blues, vmax=30, vmin=-30)
    for i in range(8):
        for j in range(8):
            c = macroblock[j,i]
            plt.text(i, j, str(c), va='center', ha='center')
    plt.colorbar()
    plt.savefig('C:\\d\\diplomarbeit\\DA\\Figures\\value_dist_dct.eps',
                format='eps', dpi=1000)
    plt.show()


def plot_dct_division():
    m = np.zeros(shape=(8, 8), dtype=np.int)
    for i in range(8):
        for j in range(8):
            if i + j < 9:
                m[i, j] = i + j + 1
            else:
                m[i, j] = 10
    for i in range(8):
        for j in range(8):
            c = m[j,i]
            plt.text(i, j, str(c), va='center', ha='center')
    
    plt.imshow(m, cmap=plt.cm.Blues, vmax=10, vmin=1)
    plt.savefig('C:\\d\\diplomarbeit\\DA\\Figures\\regions_dct_detailed.eps',
                format='eps', dpi=1000)
    plt.show()


def test_glcm_cluster():
    fmaps = np.load('./dev/fmap.npy')[0]
    fmaps = (fmaps - fmaps.min()) * 255 / (fmaps.max() - fmaps.min())
    fmaps = fmaps.astype(np.uint8).transpose((2, 0, 1))
    d = {}
    dl = []
    num_clusters = 5
    res = np.zeros(shape=(len(fmaps), 6))
    for idx, fmap in enumerate(fmaps):
        fmap = (fmap // 32).astype(np.uint8)
        g = greycomatrix(fmap, [1], [0, np.pi/4, np.pi/2,
                                3*np.pi/4], levels=8, symmetric=False, normed=True)
        contrast = greycoprops(g, 'contrast')[0][0]
        homo = greycoprops(g, 'homogeneity')[0][0]
        corr = greycoprops(g, 'correlation')[0][0]
        asm = greycoprops(g, 'ASM')[0][0]
        dis = greycoprops(g, 'dissimilarity')[0][0]
        entro = shannon_entropy(g)
        res[idx] = np.array([homo, contrast, dis, entro, asm, corr])
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, verbose=0).fit(res)
    labels = kmeans.labels_
    d_cluster = {}
    level_map = {}
    for l in range(num_clusters):
        tmp = np.where(labels==l)[0]
        fmap_tmp = 0
        sort_tmp = []
        for idx in tmp:
            sort_tmp.append((idx, np.mean(fmaps[idx])))
            fmap_tmp += fmaps[idx]
        sort_tmp = sorted(sort_tmp, key=lambda x: x[1])
        fmap_tmp = fmap_tmp / len(tmp)
        level = np.mean(fmap_tmp)
        level_map[l] = level
        d_cluster[l] = [index[0] for index in sort_tmp]
    level_map_sorted = sorted(level_map.items(), key=lambda x: x[1])
    level_map = {}
    for index, level_map_info in enumerate(level_map_sorted):
        level_map[index] = level_map_info[0]
    cnt = 0
    for i in range(num_clusters):
        for j in d_cluster[i]:
            dl.append((j, cnt))
            cnt += 1
    return dl, fmaps

if __name__ == "__main__":
    test_glcm_cluster()
