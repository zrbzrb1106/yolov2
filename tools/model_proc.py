import os
import sys
import numpy as np
import subprocess
import cv2
import time
from memory_profiler import profile

import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import Model
from keras.layers import Input
from keras import backend as K
K.set_learning_phase(0)

sys.path.append('./')
from utils.imgutils import minmax_norm, preprocess_image


def read_model(path, name):
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
    conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True
    )
    sess = tf.Session(config=conf)
    init = tf.global_variables_initializer()
    sess.run(init)
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
        tf.import_graph_def(graph_def, name=name)
    return sess


def model_split(model_path, num):
    '''
    Split model through Keras
    params:
        @model_path: path of Keras model
        @num: number of the layers to split and keep as first part
    return:
        Keras model of the first part
    '''
    layer_idx = 0
    new_model_h5_path = './model/part1_test.h5'
    model = load_model(model_path)
    new_input = Input(model.layers[0].input_shape[1:])
    new_model = new_input
    for idx, layer in enumerate(model.layers[1:]):
        new_model = layer(new_model)
        if 'conv2d' in layer.name or 'max_pooling2d' in layer.name:
            layer_idx += 1
            if layer_idx == num:
                break
    new_model = Model(new_input, new_model)
    new_model.save(new_model_h5_path)


def model_convert_keras(weights_path, cfg_path):
    '''
    *Convert weights and cfg to h5 keras model, depends on YAD2K(https://github.com/allanzelener/YAD2K)
    *model cfg and weights file can be found in:
        https://github.com/pjreddie/darknet/blob/master/cfg/yolov2.cfg 
        https://pjreddie.com/media/files/yolov2.weights
    *then run:
        ./yad2k.py yolov2.cfg yolov2.weights model/yolo.h5
    '''
    pass


def model_convert_pb(model, save_path):
    '''
    Convert h5 keras model to pb file and save
    params:
        @model: keras model path or object
        @save_path: path to save the converted model in .pb
    return:
        None
    '''
    def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
        from tensorflow.python.framework.graph_util import convert_variables_to_constants
        graph = session.graph
        with graph.as_default():
            freeze_var_names = list(
                set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
            output_names = output_names or []
            output_names += [v.op.name for v in tf.global_variables()]
            # Graph -> GraphDef ProtoBuf
            input_graph_def = graph.as_graph_def()
            if clear_devices:
                for node in input_graph_def.node:
                    node.device = ""
            frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                          output_names, freeze_var_names)
            return frozen_graph

    if type(model) == str:
        model_keras = load_model(model)
    else:
        model_keras = model
    frozen_graph = freeze_session(K.get_session(),
                                  output_names=[out.op.name for out in model_keras.outputs])
    head, tail = os.path.split(save_path)
    tf.train.write_graph(frozen_graph, head, tail, as_text=False)


@profile
def test(model_path):
    sess = read_model(model_path, '')
    layer_names = ['input_1:0', 'conv2d_1/convolution:0', 'max_pooling2d_1/MaxPool:0', 'conv2d_2/convolution:0', 
                   'max_pooling2d_2/MaxPool:0', 'conv2d_3/convolution:0', 'conv2d_4/convolution:0', 
                   'conv2d_5/convolution:0', 'max_pooling2d_3/MaxPool:0']
    input1 = sess.graph.get_tensor_by_name(layer_names[0])
    output1 = sess.graph.get_tensor_by_name(layer_names[8])

    for i in range(100):
        input_data = np.array(np.random.random_sample((1, 608, 608, 3)), dtype=np.float32)
        start = time.time()
        output_data = sess.run(sess.graph.get_tensor_by_name(layer_names[8]), feed_dict={input1:input_data})
        end = time.time()
        print("inference time: ", end-start)


def params_count():
    part1_h5_path = './model/part1_test.h5'
    model = load_model(part1_h5_path)
    model.summary()


if __name__ == "__main__":
    model_path = './model/yolo.h5'
    part1_h5_path = './model/part1_test.h5'
    out_path = './model/part1_test.pb'
    # out_path = './model/part1_test_quantized.pb'
    ### split model and save first 8 layers
    # model_split(model_path, 8)
    # K.clear_session()
    ### convert h5 model to pb
    # model_convert_pb(part1_h5_path, out_path)
    ### use graph for detect
    test(out_path)
