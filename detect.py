# coding:utf-8
######################
###  remote debug  ###
######################
# import ptvsd
# addr = ("192.168.31.222", 5678)
# ptvsd.enable_attach(address=addr, redirect_output=True)
# ptvsd.wait_for_attach()

import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import os
import time
import statistics
import sys
import io

from utils.imgutils import *
from utils.config import anchors, class_names, category_id_dict
from dev import graph_editor as ge

def get_feature_map(orig, layer, shape=None, is_display=0, is_save=0):
    batch_size, m, n, c = orig.shape
    fmap_images = []
    for batch in range(batch_size):
        if not shape:
            a = int(np.sqrt(c))
            while c % a is not 0:
                a = a - 1
            b = int(c / a)
            shape = (a, b)
        feature_maps = orig[batch]
        info_minmax = (feature_maps.min(), feature_maps.max())
        dst = np.zeros(shape=feature_maps.shape)
        feature_maps = cv2.normalize(feature_maps, dst, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        a, b = shape
        res = []
        for row in range(a):
            tmp = np.zeros(shape=(m, n, b))
            # l = cv2.normalize(feature_maps[:, :, row*b:row*b+b], tmp, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
            # l = np.round(feature_maps[:, :, row*b:row*b+b], 0).astype(np.int8)
            l = feature_maps[:, :, row*b:row*b+b]
            f_tmp = np.hstack(np.transpose(l, (2, 0, 1)))
            res.append(f_tmp)
        fmap_image = np.vstack(res)
        if is_display:
            cv2.imshow("feature", fmap_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if is_save:
            cv2.imwrite("./tmp/feature_{}.jpg".format(layer), fmap_image)
        fmap_images.append((fmap_image, info_minmax))
    return fmap_images


def eval_model(batch_size, sess):
    images_path = "pedes_images"
    imgs = [cv2.imread(os.path.join(images_path, i))
            for i in os.listdir(images_path)]

    indexes = [random.randint(0, len(imgs)) for i in range(batch_size)]
    # imgs_size = [imgs[index].shape for index in indexes]
    # generate batch
    input_size = (608, 608)
    img = preprocess_image(imgs[indexes[0]], input_size)
    print(type(img))
    for i in range(1, len(indexes)):
        img = np.append(img, preprocess_image(
            imgs[indexes[i]], input_size), axis=0)
    input_tensor = sess.graph.get_tensor_by_name("yolo/input:0")
    mid_tensor_8 = sess.graph.get_tensor_by_name("yolo/Pad_5:0")
    mid_tensor_9 = sess.graph.get_tensor_by_name("yolo/Pad_5:0")
    output_tensor = sess.graph.get_tensor_by_name("yolo/output:0")
    # feed into yolo
    t1 = []
    t2 = []
    for i in range(20):
        start = time.time()
        out1 = sess.run(mid_tensor_8, feed_dict={input_tensor: img})
        mid = time.time()
        out2 = sess.run(output_tensor, feed_dict={mid_tensor_9: out1})
        end = time.time()
        sess.run(output_tensor, feed_dict={input_tensor: img})
        cur = time.time()
        total = cur - end
        t1.append((mid - start)/batch_size)
        t2.append((end - mid)/batch_size)
        print(mid - start, end - mid, total)
    return statistics.mean(t1[2:]), statistics.mean(t2[2:]), statistics.stdev(t1[2:]), statistics.stdev(t2[2:])


def layers_feature_maps():
    input_size = (608, 608)
    img_name = "./pedes_images/01-20170320211734-25.jpg"
    img_name = "./cocoapi/images/val2017/000000569059.jpg"
    sess = ge.read_model('./model/yolo.pb', "yolo")
    img_orig = cv2.imread(img_name)
    img = preprocess_image(img_orig)
    tensor_names = [t.name for op in sess.graph.get_operations()
                    for t in op.values()]
    input_tensor = sess.graph.get_tensor_by_name("yolo/input:0")
    layers = ['0-convolutional:0', '2-maxpool:0', '3-convolutional:0', '5-maxpool:0', 
              '6-convolutional:0', '8-convolutional:0', '10-convolutional:0', '12-maxpool:0']
    filters = 'yolo/10-convolutional/filter:0'
    filters_data = sess.run(filters, feed_dict={input_tensor:img})
    np.save("./dev/filters", filters_data)
    for ind, layer in enumerate(layers):
        output_tensor = sess.graph.get_tensor_by_name("yolo/{}".format(layer))
        featuremaps = sess.run(output_tensor, feed_dict={input_tensor:img})
        get_feature_map(featuremaps, ind+1, is_display=0, is_save=1)
    

def test_feature_maps():
    input_size = (608, 608)
    img_name = "./pedes_images/01-20170320211734-25.jpg"
    sess = ge.read_model('./model/yolo.pb', "yolo")
    img_orig = cv2.imread(img_name)
    img = preprocess_image(img_orig)
    input_tensor = sess.graph.get_tensor_by_name("yolo/input:0")
    output_tensor = sess.graph.get_tensor_by_name("yolo/Pad_5:0")
    featuremaps = sess.run(output_tensor, feed_dict={input_tensor:img})

    fmaps_batch = featuremaps.transpose((0, 3, 1, 2))[0]
    np.random.shuffle(fmaps_batch)
    channels, w, h = fmaps_batch.shape

    import blosc
    import zlib
    import fpzip
    print(sys.getsizeof(copy.copy(fmaps_batch)))
    bytes_array_fmaps = fmaps_batch.tostring()
    zlib_comp = zlib.compress(bytes_array_fmaps)
    res = zlib.decompress(zlib_comp)
    res = np.frombuffer(res, dtype=np.float32)
    res = res.reshape((128, 78, 78))
    b = np.array_equal(fmaps_batch, res)
    print(0)

    # fmap_vectors = np.zeros(shape=(channels, w*h))
    # for ind, fmap in enumerate(fmaps_batch):
    #     fmap_vector = fmap.reshape((1, 78*78))[0]
    #     fmap_vectors[ind] = fmap_vector
    # fmap_vectors = (fmap_vectors - fmap_vectors.min()) * 255 / (fmap_vectors.max() - fmap_vectors.min())
    # fmap_vectors = fmap_vectors.astype(np.uint8)
    # encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
    # encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30]
    # result, encimg = cv2.imencode('.jpg', fmap_vectors, encode_param)
    # print(sys.getsizeof(encimg))
    # decimg = cv2.imdecode(encimg, 0)
    # cv2.imshow("before", fmap_vectors)
    # cv2.imshow("after", decimg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    '''
    for i in range(0, w*h):
        tmp = fmap_vectors[:, i].reshape(8, 16)
        if not tmp.max() - tmp.min() == 0:
            tmp = (tmp - tmp.min()) * 255 / (tmp.max() - tmp.min())
        tmp = tmp.astype(np.uint8)
        cv2.imshow("tmp", tmp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    '''

def detect():
    input_size = (608, 608)
    sess = ge.read_model('./model/yolo.pb', "yolo")
    # img_orig = cv2.imread('./data/car.jpg')
    
    input_tensor = sess.graph.get_tensor_by_name("yolo/input:0")
    output_tensor = sess.graph.get_tensor_by_name("yolo/output:0")
    for i in range(5):
        img_orig = cv2.imread('./pedes_images/01-20170320211734-25.jpg')
        img = preprocess_image(img_orig)
        # get feature map and if possible, display it
        # layers[num]: num is the num'th layer, e.g, layers[2] is the second layer, namely the first maxpool layer
        
        # feature_map = sess.run(sess.graph.get_tensor_by_name(
        #     "yolo/Pad_5:0"), feed_dict={input_tensor: img})
        # get_feature_map(feature_map, 0, 0)
        
        output_sizes = input_size[0]//32, input_size[1]//32
        res = sess.run(output_tensor, feed_dict={input_tensor:img})
        bboxes, obj_probs, class_probs = decode_result(model_output=res, output_sizes=output_sizes,
                                num_class=len(class_names), anchors=anchors)
        bboxes, scores, class_max_index = postprocess(
            bboxes, obj_probs, class_probs, image_shape=img_orig.shape[:2])
        
        img_detection = draw_detection(
            img_orig, bboxes, scores, class_max_index, class_names)
        # cv2.imwrite("./data/detection.jpg", img_detection)
        cv2.imshow("detection_results", img_detection)
        cv2.waitKey(0)


if __name__ == "__main__":
    # detect()
    # layers_feature_maps()
    test_feature_maps()