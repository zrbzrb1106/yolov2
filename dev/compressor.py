# coding:utf-8
######################
###  remote debug  ###
######################
# import ptvsd
# addr = ("192.168.31.222", 5678)
# ptvsd.enable_attach(address=addr, redirect_output=True)
# ptvsd.wait_for_attach()

import numpy as np
import copy
import cv2
import sys
import io
import zlib
import subprocess
# import ffmpeg
import time
import os
from PIL import Image

from dev import graph_editor as ge
from detect import *
from utils.imgutils import *
from dev.test_dct import *

# class for setting different compression methods
# for feature maps. 
class Compressor:
    def __init__(self, filters=None, k=None):
        self.buffer = []
        self.start_enc = 0
        self.end_enc = 0
        self.start_dec = 0
        self.end_dec = 0
        self.compressed_mem = 0
        self.num_pruned_values = 0
        self.k = k
        self.filters = filters
        if k is not None:
            self.labels = self.__filters_cluster(filters, k)
        self.need_re_clustering = False

    def fill_buffer(self, data, info=0):
        self.buffer = []
        compressed_data = self.compress(data, info)
        self.buffer.append(compressed_data)
        return 0

    def read_buffer(self, info=0):
        data = self.__get_from_buf()
        return self.extract(data, info)

    def compress(self, data, info):
        self.start_enc = time.time()
        # res = self.__zip_enc(data, info)
        # res = self.__float_enc(data, info)
        # res = self.__mmd_enc(data, info[0], info[1])
        # res = self.__float_split_mantissa_exponent_enc(data)
        res = self.__jpeg_enc(data, info)
        # res = self.__webp_enc(data, info)
        # res = self.__h264_enc(data, info)
        # res = self.__dct_enc(data, info)
        # res = self.__jpeg_enc_fmapwise(data, info)
        # res = self.__jpeg_enc_cluster(data, info)
        # res = self.__dct_prune_enc(data, info)
        # res = self.__space_enc(data, info)
        # res = self.__png_enc(data)
        self.end_enc = time.time()
        return res

    def extract(self, data, info=0):
        self.start_dec = time.time()
        # res = self.__zip_dec(data)
        # res = self.__float_dec(data)
        # res = self.__mmd_dec(data)
        # res = self.__float_split_mantissa_exponent_dec(data[0], data[1])
        res = self.__jpeg_dec(data)
        # res = self.__webp_dec(data)
        # res = self.__h264_dec(data)
        # res = self.__dct_dec(data)
        # res = self.__jpec_dec_fmapwise(data)
        # res = self.__jpeg_dec_cluster(data)
        # res = self.__dct_prune_dec(data)
        # res = self.__space_dec(data)
        # res = self.__png_dec(data)
        self.end_dec = time.time()
        return res

    def print_info(self):
        print(self.end_enc - self.start_enc, self.end_dec -
              self.start_dec, self.compressed_mem, self.num_pruned_values)

    def __filters_cluster(self, filters, k):
        # average filters for clustering
        # filters_processed = get_filters(filters)
        # labels = filters_clustering(filters_processed, k)
        # flatten each filter
        quant_filters = filters_quant(filters)
        labels = filters_clustering_quant(quant_filters, k)
        # np.random.shuffle(labels)
        # sort labels by position
        sorted_labels = []
        label = 0
        lut = {}
        for i in labels:
            if i not in lut.keys():
                lut[i] = label
                label += 1
            sorted_labels.append(lut[i])
        print(np.reshape(sorted_labels, (8, 16)))
        return sorted_labels

    def __get_from_buf(self):
        return self.buffer[0]

    def __zip_enc(self, data, info):
        flag = info
        data_copy = copy.copy(data)
        min_val, max_val = data_copy.min(), data_copy.max()
        if flag == 'norm':
            data_norm = minmax_norm(data_copy, 0, 255)
            data_copy = np.round(data_norm, 0).astype(np.uint8)
            compressed_array = zlib.compress(data_copy)
        else:
            compressed_array = zlib.compress(np.round(data_copy, 0).astype(np.int8))
        self.compressed_mem = len(compressed_array)
        return (compressed_array, (min_val, max_val), flag)

    def __zip_dec(self, data):
        min_val, max_val = data[1]
        flag = data[2]
        decompressed_data = zlib.decompress(data[0])
        if flag == 'norm':
            res = np.frombuffer(decompressed_data, dtype=np.uint8).reshape((78, 78, 128))
            res = minmax_norm(res, min_val, max_val)
        else:
            res = np.frombuffer(decompressed_data, dtype=np.int8).reshape((78, 78, 128))
        decompressed_array = np.zeros((1, 78, 78, 128), dtype=np.float32)
        decompressed_array[0] = res
        return decompressed_array

    def __float_enc(self, x, decimals):
        data_copy = copy.copy(x)
        if decimals <= 3:
            data_copy = np.round(data_copy, decimals).astype(np.float16)
        else:
            data_copy = data_copy.astype(np.float16)
        compressed_data = zlib.compress(data_copy)
        self.compressed_mem = len(compressed_data)
        return compressed_data

    def __float_dec(self, data):
        decompressed_data = zlib.decompress(data)
        _dtype = np.float16
        res = np.frombuffer(decompressed_data,
                            dtype=_dtype).reshape((78, 78, 128))
        decompressed_array = np.zeros((1, 78, 78, 128), dtype=np.float32)
        decompressed_array[0] = res
        return decompressed_array

    def __float_split_mantissa_exponent_enc(self, x):
        """
        x: numpy array, dtype=float
        return: mantissa and exponent of elements
        """
        data = copy.copy(x)
        print(sys.getsizeof(data))
        data = np.round(data, 1)
        mant, exp = np.frexp(data)
        exp = exp.astype(dtype=np.int8)
        a = 10
        tmp = mant * a
        mant = tmp.astype(dtype=np.int8)
        print(sys.getsizeof(mant), sys.getsizeof(exp))
        mant_zip = self.__zip_enc(mant)
        exp_zip = self.__zip_enc(exp)
        print(sys.getsizeof(mant_zip), sys.getsizeof(exp_zip))
        return mant, exp

    def __float_split_mantissa_exponent_dec(self, mant, exp):
        a = 10
        mant = mant / a
        return np.ldexp(mant, exp)

    def __h264_enc(self, x, info):
        crf = info[0]
        preset = info[1]
        method = info[2]
        feature_maps = copy.copy(x)
        min_val, max_val = feature_maps.min(), feature_maps.max()
        dl, fmaps = sort_fmaps(feature_maps, method)
        data_name = 'tmp/data'
        b = b''
        for idx, item in enumerate(dl):
            b += fmaps[item[0]].tobytes()
        with open(data_name, 'wb') as f:
            f.write(b)
            f.close()
        time.sleep(0.1)
        out, _ = (ffmpeg.input(data_name, framerate=60, f='rawvideo', s='78x78', pix_fmt='gray'). \
            output('pipe:', preset=preset,
                   pix_fmt='yuv420p', vcodec='libx264', f='h264', crf=crf). \
            run(capture_stdout=True, quiet=True))
        return (out, dl, (min_val, max_val))

    def __h264_dec(self, data):
        info = data[1]
        video_data = copy.copy(data[0])
        self.compressed_mem = len(video_data)
        min_val, max_val = data[2]
        f = open('./tmp/fmaps_tmp.h264', 'wb')
        f.write(video_data)
        f.close()
        out, _ = (ffmpeg.input('./tmp/fmaps_tmp.h264').output(
            'pipe:', f='rawvideo', vsync=0, pix_fmt='gray').run(quiet=True))
        fmaps = (np.frombuffer(out, np.uint8).reshape([-1, 78, 78, 1]))
        fmaps = (fmaps - fmaps.min()) * max_val / \
            (fmaps.max() - fmaps.min()) + min_val
        res = np.zeros(shape=(1, 78, 78, 128), dtype=np.float32)
        for idx, fmap in enumerate(fmaps):
            res[0, :, :, info[idx][0]] = fmap[:, :, 0]
        return res

    def __webp_enc(self, x, quality):
        shape = (8, 16)
        data = copy.copy(x)
        fmap_images = feature_maps_to_image(
            data, shape, is_display=0, is_save=0)
        encode_param = [int(cv2.IMWRITE_WEBP_QUALITY), quality]
        res = []
        for fmap_image in fmap_images:
            result, encimg = cv2.imencode('.webp', fmap_image[0], encode_param)
            self.compressed_mem = len(encimg)
            res.append((encimg, fmap_image[1]))
        return res

    def __webp_dec(self, data):
        shape = (8, 16)
        res_data = image_to_feature_maps(data, shape, codec='webp')
        return res_data

    def __jpeg_enc(self, x, quality):
        shape = (8, 16)
        data = copy.copy(x)
        # data = space_values_prune(data, 0.5)
        fmap_images = feature_maps_to_image(
            data, shape, is_digitize = 1, is_display=0, is_save=0)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        res = []
        for fmap_image in fmap_images:
            result, encimg = cv2.imencode('.jpg', fmap_image[0], encode_param)
            self.compressed_mem = len(encimg)
            res.append((encimg, fmap_image[1]))
        return res

    def __jpeg_dec(self, data):
        shape = (8, 16)
        res_data = image_to_feature_maps(data, shape)
        return res_data

    def __dct_enc(self, x, info):
        data = copy.copy(x)
        min_val, max_val = x.min(), x.max()
        fmap_image = get_squared_image_of_feature_maps(data, 0, (0, 1))
        height, width = fmap_image.shape
        # dct
        fmap_dct = cv2.dct(fmap_image)
        # fmap_dct = np.round(fmap_dct, 5)
        fmap_dct_int_part = np.floor(fmap_dct).astype(np.int)
        fmap_dct_f_part = (
            (np.round(fmap_dct, 2) - fmap_dct_int_part) * 100).astype(np.uint8)
        # d = calc_hist(fmap_dct_f_part)
        # fmap_dct_f_part = prune(fmap_dct_f_part, d)
        fmap_dct_int_part[800:, 800:] = 0
        fmap_dct_f_part[800:, 800:] = 0
        tmp = fmap_dct_f_part.astype(np.float32) / 100 + fmap_dct_int_part
        # t = zlib.compress(fmap_dct)
        # fmap_dct[np.abs(fmap_dct) < 0.01] = 0
        # fmap_dct[rs:, :] = 0
        # fmap_dct[150:, cs:] = 0
        res = []
        fmap_dct_processed = tmp
        res.append((fmap_dct_processed, (min_val, max_val)))
        return res

    def __dct_dec(self, data):
        fmap_dct = data[0][0]
        fmap = cv2.idct(fmap_dct)
        # cv2.imshow('fmap_idct', fmap)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        min_val, max_val = data[0][1]
        res_data = squared_fmaps_image_to_fmaps(fmap)
        res_data = (res_data - res_data.min()) * max_val / \
            (res_data.max() - res_data.min()) + min_val
        return res_data

    def __jpeg_enc_cluster(self, x, info):
        filters = info[0]
        quality = info[2]
        if info[1] != self.k:
            self.need_re_clustering = True
        labels = self.labels
        if self.need_re_clustering:
            labels = self.__filters_cluster(filters, info[1])
        feature_maps = copy.copy(x)
        batch_size, m, n, channels = feature_maps.shape
        data = np.transpose(x, (0, 3, 1, 2))
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        images_fmaps_all_batches = np.zeros((batch_size, info[1], m, n))
        encoded_info = [labels]
        for batch in range(batch_size):
            f_maps_batch = data[batch]
            res = set()
            max_ = -1
            for ind, i in enumerate(labels):
                if i > max_:
                    res.add(ind)
                    max_ = i
            f_maps = np.array([f_maps_batch[i] for i in res])
            images_fmaps_all_batches[batch] = f_maps
        fmap_images = feature_maps_to_image(
            images_fmaps_all_batches.transpose((0, 2, 3, 1)))
        for fmap_image in fmap_images:
            result, encimg = cv2.imencode('.jpg', fmap_image[0], encode_param)
            print(sys.getsizeof(encimg))
            encoded_info.append((encimg, fmap_image[1]))
        return encoded_info

    def __jpeg_dec_cluster(self, data):
        labels = data[0]
        shape = (4, 5)
        info = data[1:]
        res = np.zeros(shape=(len(info), 78, 78, 128))
        for batch, img_batch in enumerate(info):
            fmaps_raw = jpeg_img_split(img_batch[0])
            minmax = img_batch[1]
            fmaps_raw = cv2.normalize(
                fmaps_raw, fmaps_raw, minmax[0], minmax[1], cv2.NORM_MINMAX, cv2.CV_32FC1)
            for i in range(128):
                res[batch, :, :, i] = fmaps_raw[labels[i]]
        return res

    def __dct_prune_enc(self, x, info):
        feature_maps = copy.copy(x)
        fmap_img = feature_maps_to_image(feature_maps, is_digitize=0, is_unit=0)[0][0]
        # fmap_img[0:78] = 0
        # fmap_img[78:88] = 0
        # fmap_img[156:166] = 0
        fmap_img = np.expand_dims(fmap_img, 2)
        fmap_img_pad = padding(fmap_img)
        fmap_new, fmap_new_dct = dct_compress(fmap_img_pad, info)
        res = []
        min_val, max_val = feature_maps.min(), feature_maps.max()
        res.append((fmap_new, (min_val, max_val)))
        return res

    def __dct_prune_dec(self, data):
        min_val, max_val = data[0][1]
        fmap_img = data[0][0]
        rows, cols = (8, 16)
        feature_maps = np.zeros(shape=(1, 78, 78, 128))
        tmp = np.vsplit(fmap_img, rows)
        cnt = 0
        for row_data in tmp:
            row_splitted = np.hsplit(row_data, cols)
            for f_map in row_splitted:
                feature_maps[0, :, :, cnt] = f_map
                cnt += 1
        # feature_maps = (feature_maps - feature_maps.min()) * (max_val -
        #                                                       min_val) / (feature_maps.max() - feature_maps.min()) + min_val
        return feature_maps
    
    def __space_enc(self, data, thres_factor):
        data = np.round(copy.copy(data), 1).astype(np.float16)
        data_range = data.max() - data.min()
        thres = data_range / thres_factor
        res = space_values_prune(data, thres)
        self.num_pruned_values = np.count_nonzero(data) - np.count_nonzero(res)
        res = zlib.compress(res)
        self.compressed_mem = len(res)
        return res
    
    def __space_dec(self, data):
        decompressed_data = zlib.decompress(data)
        res = np.frombuffer(decompressed_data, dtype=np.float16).reshape((78, 78, 128))
        decompressed_array = np.zeros((1, 78, 78, 128), dtype=np.float32)
        decompressed_array[0] = res
        return decompressed_array

    def __png_enc(self, x):
        data = copy.copy(x)
        encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 9, int(
        cv2.IMWRITE_PNG_STRATEGY), 3]
        fmap_image = feature_maps_to_image(data, is_digitize=0)[0][0]
        min_val, max_val = fmap_image.min(), fmap_image.max()
        # fmap_image_pos = fmap_image - delta
        fmap_image_int = np.round(fmap_image).astype(np.int8)
        fmap_image_int = fmap_image_int - fmap_image_int.min()
        result, encimg_int = cv2.imencode('.png', fmap_image_int, encode_param)
        self.compressed_mem = len(encimg_int.tobytes())
        return (encimg_int, (min_val, max_val))

    def __png_dec(self, data):
        encimg = data[0]
        minmax_val = data[1]
        fmap_img = cv2.imdecode(encimg, 0)
        rows, cols = (8, 16)
        feature_maps = np.zeros(shape=(1, 78, 78, 128))
        tmp = np.vsplit(fmap_img, rows)
        cnt = 0
        for row_data in tmp:
            row_splitted = np.hsplit(row_data, cols)
            for f_map in row_splitted:
                feature_maps[0, :, :, cnt] = f_map
                cnt += 1
        feature_maps = minmax_norm(feature_maps, minmax_val[0], minmax_val[1])
        return feature_maps


if __name__ == "__main__":
    # initialize
    sess1 = ge.read_model('./model/part1.pb', "part1")
    sess2 = ge.read_model('./model/yolo.pb', "yolo")
    tensor_names = [t.name for op in sess1.graph.get_operations()
                    for t in op.values()]
    input1 = sess1.graph.get_tensor_by_name("part1/input:0")
    output1 = sess1.graph.get_tensor_by_name("part1/Pad_5:0")

    input2 = sess2.graph.get_tensor_by_name("yolo/Pad_5:0")
    output2 = sess2.graph.get_tensor_by_name("yolo/output:0")

    filters_name = 'part1/10-convolutional/filter:0'

    k = None
    # main loop
    img_orig = cv2.imread('./pedes_images/01-20170320211847-01.jpg')
    # img_orig = cv2.imread('./cocoapi/images/val2017/000000000724.jpg')
    # img_orig = cv2.imread('./cocoapi/images/val2017/000000000139.jpg')
    # img_orig = cv2.imread('./img_qt.jpg')
    img = preprocess_image(img_orig)
    filters = sess1.run(sess1.graph.get_tensor_by_name(filters_name),
                        feed_dict={input1: img})
    compressor = Compressor(filters, k)
    start = time.time()
    output_feature = sess1.run(output1, feed_dict={input1: img})
    get_feature_map(output_feature, 0)
    # feed the feature maps into the buffer and applying compresson
    flag = compressor.fill_buffer(output_feature, 90)
    if flag is not 0:
        raise "Error"
    # read the compressed feature maps and decode them
    decompressed_data = compressor.read_buffer()
    compressor.print_info()
    res = sess2.run(output2, feed_dict={input2: decompressed_data})

    bboxes, obj_probs, class_probs = decode_result(model_output=res, output_sizes=(608//32, 608//32),
                                                   num_class=len(class_names), anchors=anchors)
    bboxes, scores, class_max_index = postprocess(
        bboxes, obj_probs, class_probs, image_shape=img_orig.shape[:2])
    img_detection = draw_detection(
        img_orig, bboxes, scores, class_max_index, class_names)

    end = time.time()
    print('YOLO_v2 detection has done! spent {} seconds'.format(end - start))

    cv2.imshow("detection_results", img_detection)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
