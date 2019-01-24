# coding:utf-8

import tensorflow as tf
import numpy as np
import os
import time
import cv2
from utils import *
from detect import *

from google.protobuf import text_format
from tensorflow.python.framework import graph_io
import tensorflow.contrib.graph_editor as ge

def read_model(path):
    """
    path: the location of pb or pbtxt file path
    return: tf.Session()
    """
    # use one cpu core
    # session_conf = tf.ConfigProto(
    #   intra_op_parallelism_threads=1,
    #   inter_op_parallelism_threads=1)
    # sess = tf.Session(config=session_conf)
    # use all cores
    sess = tf.Session()
    mod = ['rb', 'r']
    flag = 'rb'
    if os.path.basename(path).split('.')[-1] == 'pb':
        flag = mod[0]
    elif os.path.basename(path).split('.')[-1] == 'pbtxt':
        flag = mod[1]
    else:
        raise "file format error"
    with tf.gfile.FastGFile(path, flag) as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name="")
    return sess


# convert pbtxt format to pb binary graph def, but memory error could happen
def convert_pbtxt_to_pb(path):
    filename = path.split("/")[-1].split(".")[0]
    with open(path, 'r') as f:
        graph_def = tf.GraphDef()
        file_content = f.read()
        text_format.Parse(file_content, graph_def)
        # text_format.Merge(file_content, graph_def)
        tf.train.write_graph(graph_def,
                             os.path.dirname(path),
                             os.path.basename(filename) + '.pb',
                             as_text=False)
    return os.path.join(os.path.dirname(path), filename+'.pb')


def split_graph(sess):
    node_names = [n.name for n in tf.get_default_graph().as_graph_def().node]
    tensor_names = [t.name for op in sess.graph.get_operations()
                    for t in op.values()]
    # save part 1
    n1 = sess.graph.get_tensor_by_name('Pad_5:0')
    sg_def = tf.graph_util.extract_sub_graph(sess.graph_def, ["Pad_5"])
    tf.train.write_graph(sg_def, "./dev/splitted_models", "part1.pb", False)
    # save part 2
    # load whole yolo.pbtxt and delete the layers we dont need, and then convert
    # it to .pb
    pass


def restore_model(path):
    """
    path: the path dir which contains both part1.pb and part2.pb
    return: both tf.Session() for part1 and part2
    """
    dirs = os.listdir(path)
    part1_models = []
    part2_models = []
    model1 = "part1.pb"
    model2 = "part2.pb"
    for m in dirs:
        if "part1" in m:
            part1_models.append(m)
        if "part2" in m:
            part2_models.append(m)
    if model1 in part1_models:
        path1 = os.path.join(path, model1)
    elif model1 + "txt" in part1_models:
        path1 = os.path.join(path, model1+"txt")
    else:
        raise "Error: No pbtxt and pb files of part1"

    if model2 in part2_models:
        path2 = os.path.join(path, model2)
    elif model2 + "txt" in part2_models:
        path2 = os.path.join(path, model2+"txt")
    else:
        raise "Error: No pbtxt and pb files of part2"

    sess1 = read_model(path1)
    sess2 = read_model(path2)
    
    return sess1, sess2


def test():
    path = './model/yolo.pb'
    sess = tf.Session()
    with tf.gfile.FastGFile(path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name="")
    tensors = [n.name for n in tf.get_default_graph().as_graph_def().node]
    print(1)
    

if __name__ == "__main__":
    # split_graph(read_model('./model/yolo.pb'))
    test()
    # initialize
    # sess1 = read_model('./model/splitted_models/part1.pb')
    # sess2 = read_model('./model/splitted_models/yolo.pb')
    # tensor_names = [t.name for op in sess1.graph.get_operations()
    #                 for t in op.values()]
    # output1 = sess1.graph.get_tensor_by_name("part1/Pad_5:0")
    # input1 = sess1.graph.get_tensor_by_name("part1/input:0")
    # # main loop
    # img_orig = cv2.imread('./pedes_images/01-20170320211734-25.jpg')
    # img = preprocess_image(img_orig)
    # output_feature = sess1.run(output1, feed_dict={input1: img})
    # # check output feature map
    # output2 = sess2.graph.get_tensor_by_name("part2/output:0")
    # input2 = sess2.graph.get_tensor_by_name("part2/input:0")
    # res = sess2.run(output2, feed_dict={input2: output_feature})

    # output_decoded = decode(model_output=output2, output_sizes=(608//32, 608//32),
    #                         num_class=len(class_names), anchors=anchors)
    # bboxes, obj_probs, class_probs = sess2.run(
    #     output_decoded, feed_dict={input2: output_feature})

    # bboxes, scores, class_max_index = postprocess(
    #     bboxes, obj_probs, class_probs, image_shape=img_orig.shape[:2])

    # img_detection = draw_detection(
    #     img_orig, bboxes, scores, class_max_index, class_names)
    # # cv2.imwrite("./data/detection.jpg", img_detection)
    # # print('YOLO_v2 detection has done!')
    # cv2.imshow("detection_results", img_detection)
    # cv2.waitKey(0)

    # get_feature_map(output_feature, 1)
    # print(0)