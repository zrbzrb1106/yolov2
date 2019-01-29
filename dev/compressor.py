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

from dev import graph_editor as ge
from detect import *
from utils.imgutils import *


class Compressor:
    def __init__(self):
        self.buffer = []

    def fill_buffer(self, data):
        self.buffer = []
        compressed_data = self.compress(data)
        self.buffer.append(compressed_data)
        return 0

    def read_buffer(self):
        data = self.__get_from_buf()
        return self.extract(data)

    def compress(self, data):
        res = self.__zip_enc(data)
        # res = self.__float_split_mantissa_exponent_enc(data)
        return res

    def extract(self, data):
        res = self.__zip_dec(data)
        # res = self.__float_split_mantissa_exponent_dec(data[0], data[1])
        return res

    def __get_from_buf(self):
        return self.buffer[0]

    def __zip_enc(self, data):
        data_copy = copy.copy(data)
        # print(sys.getsizeof(data_copy))
        # data_copy = data_copy.astype(dtype=np.int8)
        # print(sys.getsizeof(data_copy))
        compressed_array = io.BytesIO()
        np.savez_compressed(compressed_array, data_copy)
        # print(sys.getsizeof(compressed_array))
        return compressed_array

    def __zip_dec(self, data):
        data.seek(0)
        decompressed_array = np.load(data)['arr_0']
        return decompressed_array

    def __float_split_mantissa_exponent_enc(self, x):
        """
        x: numpy array, dtype=float
        return: mantissa and exponent of elements
        """
        data = copy.copy(x)
        print(sys.getsizeof(data))
        mant, exp = np.frexp(data)
        exp = exp.astype(dtype=np.int8)
        a = 1000
        tmp = mant * a
        mant = tmp.astype(dtype=np.int16)
        print(sys.getsizeof(mant), sys.getsizeof(exp))
        mant_zip = self.__zip_enc(mant)
        exp_zip = self.__zip_enc(exp)
        print(sys.getsizeof(mant_zip), sys.getsizeof(exp_zip))
        return mant, exp

    def __float_split_mantissa_exponent_dec(self, mant, exp):
        a = 1000
        mant = mant / a
        return np.ldexp(mant, exp)


if __name__ == "__main__":
    # initialize
    sess1 = ge.read_model('./model/splitted_models/part1.pb', "part1")
    sess2 = ge.read_model('./model/splitted_models/yolo.pb', "yolo")
    tensor_names = [t.name for op in sess1.graph.get_operations()
                    for t in op.values()]
    input1 = sess1.graph.get_tensor_by_name("part1/input:0")
    output1 = sess1.graph.get_tensor_by_name("part1/Pad_5:0")

    input2 = sess2.graph.get_tensor_by_name("yolo/Pad_5:0")
    output2 = sess2.graph.get_tensor_by_name("yolo/output:0")

    # input3 = sess2.graph.get_tensor_by_name("yolo/input:0")
    # output3 = sess2.graph.get_tensor_by_name("yolo/output:0")
    compressor = Compressor()
    # main loop
    img_orig = cv2.imread('./pedes_images/01-20170320211735-19.jpg')
    img_orig = cv2.imread('./cocoapi/images/val2017/000000000724.jpg')
    img = preprocess_image(img_orig)
    start = time.time()
    output_feature = sess1.run(output1, feed_dict={input1: img})
    get_feature_map(output_feature, 0)
    s = time.time()
    flag = compressor.fill_buffer(output_feature)
    print("time for compression: ", time.time() - s)
    if flag is not 0:
        raise "Error"
    s = time.time()
    decompressed_data = compressor.read_buffer()
    print("time for decompression: ", time.time() - s)
    # decompressed_data = np.load("./dev/data/feature_compressed.npy")
    res = sess2.run(output2, feed_dict={input2: decompressed_data})

    bboxes, obj_probs, class_probs = decode_result(model_output=res, output_sizes=(608//32, 608//32),
                                                   num_class=len(class_names), anchors=anchors)

    bboxes, scores, class_max_index = postprocess(
        bboxes, obj_probs, class_probs, image_shape=img_orig.shape[:2])

    img_detection = draw_detection(
        img_orig, bboxes, scores, class_max_index, class_names)
    # cv2.imwrite("./data/detection.jpg", img_detection)
    end = time.time()
    print('YOLO_v2 detection has done! spent {} seconds'.format(end - start))

    cv2.imshow("detection_results", img_detection)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
