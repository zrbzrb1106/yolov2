# testing the compression performance of 
# different methods for feature maps
# by generating the results based on COCO
# Dataset

import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import os
import time
import json
import sys

from detect import *
from utils.imgutils import *
from utils.config import anchors, class_names, category_id_dict
from dev import graph_editor as ge
from dev import compressor


def init_model():
    # init model
    sess1 = ge.read_model('./model/part1.pb', "part1")
    sess2 = ge.read_model('./model/yolo.pb', "yolo")
    sesses = [sess1, sess2]
    return sesses

    
def fmaps_space_analysis(featuremaps):
    fmaps = copy.copy(featuremaps)
    factors = [100, 50, 20, 10, 5]
    r = fmaps.max() - fmaps.min()
    nums = []
    for f in factors:
        thres = r / f
        fmaps_tmp = copy.copy(fmaps)
        fmaps_tmp[np.abs(fmaps) < thres] = 0
        nums.append(78*78*128-np.count_nonzero(fmaps_tmp))
    print(78*78*128, nums[0], nums[1], nums[2], nums[3], nums[4])


def save_fmaps(fmap, cnt):
    fmaps = copy.copy(fmap)
    fmaps = minmax_norm(fmaps, 0, 255).astype(np.uint8)[0]
    fmaps = fmaps.transpose((2, 0, 1))
    scene_num = cnt // 200
    frame_num = cnt % 200
    channel_num = 0
    base = './channel_images/'
    for fmap in fmaps:
        filename = base + 'fmap_scene_{}_frame_{}_channel_{}.jpg'.format(scene_num, frame_num, channel_num)
        cv2.imwrite(filename, fmap)
        channel_num += 1
    print("Current frame {} finished".format(cnt))


def gen_results(sesses, imgs_path, info=0):
    """
    detect object of all images and generate the results format according to coco submission
    @param:
        sesses: a list of sessions
        imgs_path: image dataset path
        info: a tuple
            info[0]: parameter for compression, e.g. JPEG compression quality
            info[1]: 0: only do the first part yolo inference without saving the results
                     1: both inference and save the results for COCO dataset testing
    """

    # init model
    sess1, sess2 = sesses

    input1 = sess1.graph.get_tensor_by_name("part1/input:0")
    output1 = sess1.graph.get_tensor_by_name("part1/Pad_5:0")

    input2 = sess2.graph.get_tensor_by_name("yolo/Pad_5:0")
    output2 = sess2.graph.get_tensor_by_name("yolo/output:0")

    compressor_obj = compressor.Compressor(None, None)
    # read images
    base = imgs_path
    img_names = sorted(os.listdir(imgs_path))
    img_paths = [os.path.join(base, img_name) for img_name in img_names]
    # generate results
    keys = ["image_id", "category_id", "bbox", "score"]
    results = []
    input_size = (608, 608)
    cnt = 0
    for img_name in img_paths:
        if info[1] == 1:
            img_id = int(img_name.split("/")[-1].split(".")[0])
        img_orig = cv2.imread(img_name)
        img = preprocess_image(img_orig)
        output_sizes = input_size[0]//32, input_size[1]//32
        featuremap = sess1.run(output1, feed_dict={input1:img})
        # Fill buffer
        flag = compressor_obj.fill_buffer(featuremap, info[0])
        cnt += 1
        # if flag is not 0:
        #     raise "Error"
        # Read buffer
        decompressed_data = compressor_obj.read_buffer(info)
        if info[1] == 0:
            compressor_obj.print_info()
        if info[1] == 1: 
            model_output = sess2.run(output2, feed_dict={input2: decompressed_data})
            bboxes, obj_probs, class_probs = decode_result(model_output=model_output, output_sizes=output_sizes,
                                    num_class=len(class_names), anchors=anchors)
            bboxes, scores, class_max_index = postprocess(
                bboxes, obj_probs, class_probs, image_shape=img_orig.shape[:2])
            img_detection = draw_detection(
                img_orig, bboxes, scores, class_max_index, class_names)
            # cv2.imshow("detection_results", img_detection)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            for i in range(len(bboxes)):
                res = {}
                res[keys[0]] = img_id
                res[keys[1]] = int(category_id_dict[class_names[class_max_index[i]]])
                res[keys[2]] = [int(j) for j in [bboxes[i][0], bboxes[i][1], bboxes[i]
                                [2] - bboxes[i][0], bboxes[i][3] - bboxes[i][1]]]
                res[keys[3]] = round(float(scores[i]), 3)
                results.append(res)
    return results

def gen_results_jpeg():
    qual = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    savedStdout = sys.stdout
    sesses = init_model()
    for q in qual:
        with open('./logs_pedes/log_jpeg_q_{}.log'.format(q), 'w') as file:
            sys.stdout = file
            results = gen_results(sesses, './pedes_images', (q, 0))
            sys.stdout = savedStdout
        # with open('./cocoapi/results/results_jpeg_q_{}.json'.format(q), 'w') as outfile:
        #     json.dump(results, outfile)
        print("q = {} finished!".format(q))
    print("All finished!")

def gen_results_webp():
    qual = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    savedStdout = sys.stdout
    sesses = init_model()
    for q in qual:
        with open('./logs_pedes/log_webp_q_{}.log'.format(q), 'w') as file:
            sys.stdout = file
            results = gen_results(sesses, './pedes_images', (q, 0))
            sys.stdout = savedStdout
        # with open('./cocoapi/results/results_webp_q_{}.json'.format(q), 'w') as outfile:
        #     json.dump(results, outfile)
        print("q = {} finished!".format(q))
    print("All finished!")

def gen_results_h264():
    crfs = [15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35]
    savedStdout = sys.stdout
    sesses = init_model()
    for crf in crfs:
        with open('./logs/log_delta_h264_crf_{}.log'.format(crf), 'w') as file:
            sys.stdout = file
            results = gen_results(sesses, './cocoapi/images/val2017', crf)
            sys.stdout = savedStdout
        with open('./cocoapi/results/results_delta_h264_crf_{}.json'.format(crf), 'w') as outfile:
            json.dump(results, outfile)
        print("crf = {} finished!".format(crf))
    print("All finished!")

def gen_results_cut():
    cuts = ['orig', 'norm']
    sesses = init_model()
    savedStdout = sys.stdout
    for cut in cuts:
        # results = gen_results(sesses, './cocoapi/images/val2017', (cut, 1))
        # with open('./cocoapi/results/results_cut_{}.json'.format(cut), 'w') as outfile:
        #     json.dump(results, outfile)
        with open('./logs_pedes/log_cut_{}.log'.format(cut), 'w') as file:
            sys.stdout = file
            gen_results(sesses, './pedes_images', (cut, 0))
            sys.stdout = savedStdout
    print("Finished!")


def gen_results_prune():
    # prune_factors = [180, 160, 140, 120, 100, 80, 60, 40, 20, 10, 5]
    prune_factors = [40]
    sesses = init_model()
    savedStdout = sys.stdout
    for factor in prune_factors:
        with open('./logs_pedes/log_prune_factor_{}_1_decimal.log'.format(factor), 'w') as file:
            sys.stdout = file
            results = gen_results(sesses, './pedes_images', (factor, 0))
            sys.stdout = savedStdout
        # results = gen_results(sesses, './cocoapi/images/val2017', (factor, 1))
        # with open('./cocoapi/results/results_prune_factor_{}.json'.format(factor), 'w') as outfile:
        #     json.dump(results, outfile)
    print("Finished!")


def gen_results_float():
    decimals = [10, 3, 2, 1]
    sesses = init_model()
    savedStdout = sys.stdout
    for num_decimals in decimals:
        with open('./logs_pedes/log_float_decimals_{}.log'.format(num_decimals), 'w') as file:
            sys.stdout = file
            gen_results(sesses, './pedes_images', (num_decimals, 0))
            sys.stdout = savedStdout
        # with open('./cocoapi/results/results_float_decimals_{}.json'.format(num_decimals), 'w') as outfile:
        #     json.dump(results, outfile)
    print("Finished!")


def gen_results_quant():
    quants = [10, 30, 50, 70, 90, 110, 130][-1::-1]
    sesses = init_model()
    savedStdout = sys.stdout
    for quant in quants:
        with open('./logs/log_quant_{}.log'.format(quant), 'w') as file:
            sys.stdout = file
            print("quant=={} 3115125 778864".format(quant))
            results = gen_results(sesses, './cocoapi/images/val2017', quant)
            sys.stdout = savedStdout
        with open('./cocoapi/results/results_quant_{}.json'.format(quant), 'w') as outfile:
            json.dump(results, outfile)


def gen_results_h264_all_preset():
    crfs = [20, 22, 24, 26, 28, 30]
    presets = ['slower', 'medium', 'faster', 'ultrafast']
    savedStdout = sys.stdout
    sesses = init_model()
    for preset in presets:
        for crf in crfs:
            with open('./logs_pedes/log_h264_preset_{}_crf_{}.log'.format(preset, crf), 'w') as file:
                sys.stdout = file
                results = gen_results(sesses, './pedes_images', ((crf, preset), 0))
                sys.stdout = savedStdout
            with open('./cocoapi/results/results_h264_preset_{}_crf_{}.json'.format(preset, crf), 'w') as outfile:
                json.dump(results, outfile)
            print("crf = {} finished!".format(crf))
        print("preset={} finished!".format(preset))
    print("All finished!")


def gen_results_png():
    savedStdout = sys.stdout
    sesses = init_model()
    with open('./logs_pedes/log_png.log', 'w') as file:
        sys.stdout = file
        results = gen_results(sesses, './pedes_images', (10, 0))
        sys.stdout = savedStdout
    print('png finished')


def gen_results_dct_prune():
    outs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    sesses = init_model()
    savedStdout = sys.stdout
    for out in outs:
        with open('./logs/log_dct_prune_{}.log'.format(out), 'w') as file:
            sys.stdout = file
            results = gen_results(sesses, './cocoapi/images/val2017', out)
            sys.stdout = savedStdout
        with open('./cocoapi/results/results_dct_prune_{}.json'.format(out), 'w') as outfile:
            json.dump(results, outfile)


def gen_results_fmaps_space_analysis():
    sesses = init_model()
    savedStdout = sys.stdout
    with open('logs_pedes/log_space_analysis_coco.log', 'w') as file:
        sys.stdout = file
        gen_results(sesses, './cocoapi/images/val2017', (10, 0))
    print("All Finished!")


def gen_results_fmaps_sorting_test():
    crfs = [20, 25]
    methods = ['random', 'lum', 'glcm']
    sesses = init_model()
    savedStdout = sys.stdout
    for crf in crfs:
        for method in methods:
            with open('logs_pedes/log_h264_sort_{}_crf_{}.log'.format(method, crf), 'w') as file:
                sys.stdout = file
                gen_results(sesses, './pedes_images', ((crf, 'medium', method), 0))
                sys.stdout = savedStdout
            results = gen_results(sesses, './cocoapi/images/val2017', ((crf, 'medium', method), 1))
            with open('./cocoapi/results/results_h264_sort_{}_crf_{}.json'.format(method, crf), 'w') as outfile:
                json.dump(results, outfile)


if __name__ == "__main__":

    """
    before testing, coco dataset api should be installed
    see https://github.com/cocodataset/cocoapi http://cocodataset.org/#home
    *val2017 images are used in this repo, so just download it and put in ./cocoapi/images/val2017/
    *results will be save in ./cocoapi/results
    """

    # gen_results_cut()
    # gen_results_float()
    gen_results_jpeg()
    # gen_results_quant()
    # gen_results_webp()
    # gen_results_h264()
    # gen_results_h264_all_preset()
    # gen_results_dct_prune()
    # gen_results_prune()
    # gen_results_png()
    # gen_results_fmaps_space_analysis()
    # gen_results_fmaps_sorting_test()
