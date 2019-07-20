# coding:utf-8
import cv2
import tensorflow as tf
import numpy as np
import json

import socket
import sys
import os
import struct
import copy
import time

sys.path.append('./')
from utils.imgutils import *
from utils import config


class Server:
    def __init__(self):
        self.__name = 'yolo'
        self.__model_path = './model/yolo.pb'
        self.__input_tensor_name = 'Pad_5:0'
        self.__output_tensor_name = 'output:0'
        self.sess = self.__read_model(self.__model_path, self.__name, is_onecore=False)
        self.input1 = self.sess.graph.get_tensor_by_name('{}/{}'.format(self.__name, self.__input_tensor_name))
        self.output1 = self.sess.graph.get_tensor_by_name('{}/{}'.format(self.__name, self.__output_tensor_name))
        self.dtype_header = np.float16
        self.dtype_payload = np.uint8
        self.shape_splice = (8, 16)

    def __read_model(self, path, name, is_onecore=True):
        """
        path: the location of pb file path
        name: name of tf graph
        return: tf.Session()
        """
        sess = tf.Session()
        # use one cpu core
        if is_onecore:
            session_conf = tf.ConfigProto(
                intra_op_parallelism_threads=1,
                inter_op_parallelism_threads=1)
            sess = tf.Session(config=session_conf)
        
        mode = 'rb'
        with tf.gfile.FastGFile(path, mode) as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name=name)
        return sess

    def server_decode(self, data):
        """
        @header: (first 8 bytes are header)
        h_1: batch size (1 bytes)
        h_2: mode (0: JPEG  1: WebP) (1 bytes)
        l1: length feature maps info header (maximal and mininal values of the feature maps) (2 bytes)
        lp: length of payload (encoded feature maps) (4 bytes)
        @data
        header_tmp: feature maps info header
        payload_tmp: encoded feature maps
        """
        # header parsing
        h_1 = data[0]
        h_2 = data[1]
        l1 = struct.unpack('<H', data[2:4])[0]
        lp = struct.unpack('>I', data[4:8])[0]
        header_tmp = np.frombuffer(data[8:8+l1], self.dtype_header)
        payload_tmp = np.frombuffer(data[8+l1:], self.dtype_payload)
        # predict
        codec = 'jpg' if h_2 == 0 else 'webp'
        feature_maps = image_to_feature_maps([(payload_tmp, header_tmp)], self.shape_splice)
        res = self.sess.run(self.output1, feed_dict={self.input1:feature_maps})
        bboxes, obj_probs, class_probs = decode_result(model_output=res, output_sizes=(608//32, 608//32),
                                                   num_class=len(class_names), anchors=anchors)
        # image_shape: original image size for displaying
        bboxes, scores, class_max_index = postprocess(
            bboxes, obj_probs, class_probs, image_shape=(432, 320))
        # draw detection on original image (the jpg input image)
        img_detection = draw_detection(
            img_orig, bboxes, scores, class_max_index, class_names)

        return bboxes, scores, class_max_index, class_names
