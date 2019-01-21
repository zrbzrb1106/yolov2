import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import os
import time
import json
from detect import *

from utils import *
from config import anchors, class_names


def gen_results(imgs_path):
    """
    detect object of all images and generate the results according to coco submission
    """
    # init model
    sess, layers = sess, layers = read_model_pb_into_session('./model/yolo.pb')
    input_tensor = sess.graph.get_tensor_by_name(layers[0])
    output_tensor = sess.graph.get_tensor_by_name(layers[-1])
    # read images
    base = imgs_path
    img_names = sorted(os.listdir(imgs_path))
    img_paths = [os.path.join(base, img_name) for img_name in img_names][0:10]
    # generate results
    input_size = (608, 608)
    keys = ["image_id", "category_id", "bbox", "score"]
    results = []
    for img_name in img_paths:
        img_id = int(img_name.split("/")[-1].split(".")[0])
        img_orig = cv2.imread(img_name)
        img = preprocess_image(img_orig)
        output_sizes = input_size[0]//32, input_size[1]//32
        output_decoded = decode(model_output=output_tensor, output_sizes=output_sizes,
                                num_class=len(class_names), anchors=anchors)
        bboxes, obj_probs, class_probs = sess.run(
            output_decoded, feed_dict={input_tensor: img})

        bboxes, scores, class_max_index = postprocess(
            bboxes, obj_probs, class_probs, image_shape=img_orig.shape[:2])
        for i in range(len(bboxes)):
            res = {}
            res[keys[0]] = img_id
            res[keys[1]] = int(class_max_index[i])
            res[keys[2]] = [int(i) for i in [bboxes[i][0], bboxes[i][1], bboxes[i]
                            [2] - bboxes[i][0], bboxes[i][3] - bboxes[i][1]]]
            res[keys[3]] = round(float(scores[i]), 3)
            results.append(res)
    return results


if __name__ == "__main__":

    results = gen_results('/home/zrb/apps/cocoapi/images/val2017')
    print(results)
    with open('./dev/results.json', 'w') as outfile:
        json.dump(results, outfile)

