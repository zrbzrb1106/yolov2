import tensorflow as tf
import numpy as np
import cv2
import colorsys
import random
import copy
import io
import time
import matplotlib.pyplot as plt
from PIL import Image

from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize, scale, minmax_scale
from sklearn.decomposition import PCA

from skimage.feature import greycoprops, greycomatrix
from skimage.measure import shannon_entropy

def preprocess_image(image, image_size=(608, 608)):

    image_cp = np.copy(image).astype(np.float32)

    # resize image
    image_rgb = cv2.cvtColor(image_cp, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, image_size)

    # normalize
    image_normalized = image_resized.astype(np.float32) / 225.0

    # expand dimension
    image_expanded = np.expand_dims(image_normalized, axis=0)

    return image_expanded


def postprocess(bboxes, obj_probs, class_probs, image_shape=(608, 608), threshold=0.5):

    bboxes = np.reshape(bboxes, [-1, 4])

    bboxes[:, 0:1] *= float(image_shape[1])  # xmin*width
    bboxes[:, 1:2] *= float(image_shape[0])  # ymin*height
    bboxes[:, 2:3] *= float(image_shape[1])  # xmax*width
    bboxes[:, 3:4] *= float(image_shape[0])  # ymax*height
    bboxes = bboxes.astype(np.int32)

    bbox_min_max = [0, 0, image_shape[1]-1, image_shape[0]-1]
    bboxes = bboxes_cut(bbox_min_max, bboxes)

    obj_probs = np.reshape(obj_probs, [-1])
    class_probs = np.reshape(class_probs, [len(obj_probs), -1])
    class_max_index = np.argmax(class_probs, axis=1)  # 得到max类别概率对应的维度
    class_probs = class_probs[np.arange(len(obj_probs)), class_max_index]
    scores = obj_probs * class_probs

    keep_index = scores > threshold
    class_max_index = class_max_index[keep_index]
    scores = scores[keep_index]
    bboxes = bboxes[keep_index]

    class_max_index, scores, bboxes = bboxes_sort(
        class_max_index, scores, bboxes)

    class_max_index, scores, bboxes = bboxes_nms(
        class_max_index, scores, bboxes)

    return bboxes, scores, class_max_index


def draw_detection(im, bboxes, scores, cls_inds, labels, thr=0.3):
    # Generate colors for drawing bounding boxes.
    hsv_tuples = [(x/float(len(labels)), 1., 1.) for x in range(len(labels))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    # draw image
    imgcv = np.copy(im)
    h, w, _ = imgcv.shape
    for i, box in enumerate(bboxes):
        if scores[i] < thr:
            continue
        cls_indx = cls_inds[i]

        thick = int((h + w) / 300)
        cv2.rectangle(imgcv, (box[0], box[1]),
                      (box[2], box[3]), colors[cls_indx], thick)
        mess = '%s: %.3f' % (labels[cls_indx], scores[i])
        if box[1] < 20:
            text_loc = (box[0] + 2, box[1] + 15)
        else:
            text_loc = (box[0], box[1] - 10)
        # cv2.rectangle(imgcv, (box[0], box[1]-20), ((box[0]+box[2])//3+120, box[1]-8), (125, 125, 125), -1)  # puttext函数的背景
        cv2.putText(imgcv, mess, text_loc, cv2.FONT_HERSHEY_SIMPLEX,
                    1e-3*h, (255, 255, 255), thick//3)
    return imgcv


def bboxes_cut(bbox_min_max, bboxes):
    bboxes = np.copy(bboxes)
    bboxes = np.transpose(bboxes)
    bbox_min_max = np.transpose(bbox_min_max)
    # cut the box
    bboxes[0] = np.maximum(bboxes[0], bbox_min_max[0])  # xmin
    bboxes[1] = np.maximum(bboxes[1], bbox_min_max[1])  # ymin
    bboxes[2] = np.minimum(bboxes[2], bbox_min_max[2])  # xmax
    bboxes[3] = np.minimum(bboxes[3], bbox_min_max[3])  # ymax
    bboxes = np.transpose(bboxes)
    return bboxes


def bboxes_sort(classes, scores, bboxes, top_k=400):
    index = np.argsort(-scores)
    classes = classes[index][:top_k]
    scores = scores[index][:top_k]
    bboxes = bboxes[index][:top_k]
    return classes, scores, bboxes

# calculate IOU


def bboxes_iou(bboxes1, bboxes2):
    bboxes1 = np.transpose(bboxes1)
    bboxes2 = np.transpose(bboxes2)

    int_ymin = np.maximum(bboxes1[0], bboxes2[0])
    int_xmin = np.maximum(bboxes1[1], bboxes2[1])
    int_ymax = np.minimum(bboxes1[2], bboxes2[2])
    int_xmax = np.minimum(bboxes1[3], bboxes2[3])

    int_h = np.maximum(int_ymax-int_ymin, 0.)
    int_w = np.maximum(int_xmax-int_xmin, 0.)

    int_vol = int_h * int_w
    vol1 = (bboxes1[2] - bboxes1[0]) * (bboxes1[3] - bboxes1[1])
    vol2 = (bboxes2[2] - bboxes2[0]) * (bboxes2[3] - bboxes2[1])
    IOU = int_vol / (vol1 + vol2 - int_vol)
    return IOU


def bboxes_nms(classes, scores, bboxes, nms_threshold=0.5):
    keep_bboxes = np.ones(scores.shape, dtype=np.bool)
    for i in range(scores.size-1):
        if keep_bboxes[i]:
            # Computer overlap with bboxes which are following.
            overlap = bboxes_iou(bboxes[i], bboxes[(i+1):])
            # Overlap threshold for keeping + checking part of the same class
            keep_overlap = np.logical_or(
                overlap < nms_threshold, classes[(i+1):] != classes[i])
            keep_bboxes[(i+1):] = np.logical_and(keep_bboxes[(i+1):],
                                                 keep_overlap)

    idxes = np.where(keep_bboxes)
    return classes[idxes], scores[idxes], bboxes[idxes]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def decode_result(model_output, output_sizes = (19, 19), num_class=80, anchors=None):
    H, W = output_sizes
    num_anchors = len(anchors)
    anchors = np.array(anchors, dtype=np.float)
    # -1, 19*19, 5, 85
    detection_result = np.reshape(model_output, (-1, H*W, num_anchors, num_class+5))
    xy_offset = sigmoid(detection_result[:, :, :, 0:2])
    wh_offset = np.exp(detection_result[:,:,:,2:4])
    obj_probs = sigmoid(detection_result[:,:,:,4])
    class_probs = softmax(detection_result[:,:,:,5:])
    height_index = np.arange(H, dtype=np.float)
    width_index = np.arange(W, dtype=np.float)
    x_cell, y_cell = np.meshgrid(height_index, width_index)
    x_cell = np.reshape(x_cell, [1, -1, 1])
    y_cell = np.reshape(y_cell, [1, -1, 1])
    # decode
    bbox_x = (x_cell + xy_offset[:, :, :, 0]) / W
    bbox_y = (y_cell + xy_offset[:, :, :, 1]) / H
    bbox_w = (anchors[:, 0] * wh_offset[:, :, :, 0]) / W
    bbox_h = (anchors[:, 1] * wh_offset[:, :, :, 1]) / H
    bboxes = np.stack([bbox_x-bbox_w/2, bbox_y-bbox_h/2,
                       bbox_x+bbox_w/2, bbox_y+bbox_h/2], axis=3)

    return bboxes, obj_probs, class_probs


def minmax_norm(arr, minv, maxv):
    return (arr - arr.min()) * \
        (maxv - minv) / (arr.max() -
                arr.min()) + minv


def feature_maps_to_image(orig, shape=None, is_digitize = 1, is_unit = 0, is_display=0, is_save=0):
    """
    orig: original feature maps tensor
    shape: the shape of sliced image
    return: sliced images of feature maps with their min and max info when scaling 
    """
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
        if is_digitize:
            feature_maps = minmax_norm(feature_maps, 0, 255)
            feature_maps = feature_maps.astype(np.uint8)
        if is_unit:
            feature_maps = minmax_norm(feature_maps, 0, 1)
            feature_maps = feature_maps.astype(np.float)
        a, b = shape
        res = []
        for row in range(a):
            l = feature_maps[:, :, row*b:row*b+b]
            f_tmp = np.hstack(np.transpose(l, (2, 0, 1)))
            res.append(f_tmp)
        fmap_image = np.vstack(res)
        if is_display:
            cv2.imshow("fmap", fmap_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if is_save:
            cv2.imwrite("C:\\d\\diplomarbeit\\DA\\Figures\\fmap8.jpg", fmap_image)
        fmap_images.append((fmap_image, info_minmax))
    return fmap_images


def image_to_feature_maps(images_info, shape=None, codec='jpg'):
    '''
    images: array of jpeg images and it's matadata in batch
    shape:  rows and columns of feature maps allocation in jpeg image, 
            (a, b): a rows and b columns feature maps in one jpeg image
    '''
    batch_size = len(images_info)
    if not shape:
        print("Shape is not provided")
    rows, cols = shape
    feature_maps = np.zeros(shape=(batch_size, 78, 78, 128))
    cnt = 0
    decoded_img = 0
    for batch in range(batch_size):
        if codec == 'jpg':
            decoded_img = cv2.imdecode(images_info[batch][0], 0)
        elif codec == 'dct':
            decoded_img = cv2.idct(images_info[batch][0])
        elif codec == 'webp':
            decoded_img = cv2.imdecode(images_info[batch][0], 0)
        minmax_info = images_info[batch][1]
        tmp = np.zeros(shape=decoded_img.shape)
        decoded_img = cv2.normalize(decoded_img, tmp, minmax_info[0], minmax_info[1], cv2.NORM_MINMAX, cv2.CV_32FC1)
        tmp = np.vsplit(decoded_img, rows)
        for row_data in tmp:
            row_splitted = np.hsplit(row_data, cols)
            for f_map in row_splitted:
                feature_maps[batch, :, :, cnt] = f_map
                cnt += 1
        # assert cnt == 128
        cnt = 0
    return feature_maps


def get_squared_image_of_feature_maps(fmaps, is_show=0, norm_range=None):
    feature_maps = copy.copy(fmaps)
    if norm_range:
        feature_maps = (feature_maps - feature_maps.min()) * \
            norm_range[1] / (feature_maps.max() -
                             feature_maps.min()) + norm_range[0]
    batch_size, width, height, channels = feature_maps.shape
    # find proper lenth to get squared feature map
    base = channels
    while(1):
        if np.power(int(np.sqrt(base)), 2) == base:
            break
        base += 1
    num = int(np.sqrt(base))
    # get squared image sliced by feature maps
    shape_slice = (num, num)
    # assert batchsize == 1
    fmap_batch = feature_maps[0]
    # fmap_batch_normalized = (fmap_batch - fmap_batch.min()) * \
    #     1 / (fmap_batch.max() - fmap_batch.min()) + 0
    res = np.zeros(shape=(num*width, num*height))
    fmap_batch = np.transpose(fmap_batch, (2, 0, 1))
    num_rest = channels
    for row in range(num):
        if num_rest == 0:
            break
        if num_rest > num:
            l = fmap_batch[row*num:row*num+num, :, :]
            num_rest -= num
        else:
            l = fmap_batch[row*num:row*num+num_rest, :, :]
            zeros = np.zeros(shape=(num-num_rest, width, height))
            l = np.concatenate((l, zeros), axis=0)
            num_rest = 0
        f_tmp = np.hstack(l)
        res[row*height:(row*height+height)] = f_tmp
    if is_show:
        cv2.imshow("fmap", res)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return res


def squared_fmaps_image_to_fmaps(fmaps_img, shape=(1, 78, 78, 128)):
    fmaps_img = copy.copy(fmaps_img)
    height, width = fmaps_img.shape
    feature_maps = np.zeros(shape=shape, dtype=np.float32)
    s = 0
    rest = shape[3]
    rows = int(height / shape[1])
    cols = int(width / shape[2])
    rows_data = np.vsplit(fmaps_img, rows)
    for idx, row in enumerate(rows_data):
        s = idx*cols
        e = idx*cols + cols
        tmp = np.hsplit(row, cols)
        if rest < cols:
            feature_maps[0, :, :, s: s +
                         rest] = np.array(tmp[0:rest]).transpose((1, 2, 0))
            break
        else:
            feature_maps[0, :, :, s: e] = np.array(
                tmp[0: cols]).transpose((1, 2, 0))
        rest -= (e - s)
    return feature_maps


def jpeg_img_split(jpeg_img):
    m, n = (78, 78)
    jpeg_img_cp = copy.copy(jpeg_img)
    decoded_img = cv2.imdecode(jpeg_img_cp, 0)
    w, h = decoded_img.shape
    rows = int(w / m)
    cols = int(h / n)
    res = np.zeros(shape=(rows*cols, m, n))
    tmp = np.vsplit(decoded_img, rows)
    cnt = 0
    for row_data in tmp:
        row_splitted = np.hsplit(row_data, cols)
        for f_map in row_splitted:
            res[cnt] = f_map
            cnt += 1
    return res


# filters related funcs
def get_filters(filters):
    filters_data = copy.copy(filters)
    filters_data = np.transpose(filters_data, (3, 2, 0, 1))
    out_channels, in_channels, h, w = filters_data.shape
    res = np.zeros(shape=(out_channels, h, w))
    # get average of layers in filter
    for index, filt in enumerate(filters_data):
        dst = np.zeros(shape=(h, w))
        for cur in filt:
            dst = dst + cur
        dst = dst / in_channels
        res[index] = dst
    res_normalized = (res - res.min()) * 255 / (res.max() - res.min()) + 0
    return res_normalized


def filters_clustering(filters, n_clusters=0):
    filters_copy = copy.copy(filters)
    num_filters, h, w = filters_copy.shape
    filter_vecs = np.zeros(shape=(num_filters, h*w), dtype=np.uint8)
    for i in range(num_filters):
        filter_vecs[i] = filters_copy[i].reshape((1, h*w))
    if n_clusters == 0:
        return 0
    kmeans = KMeans(n_clusters=n_clusters, verbose=0).fit(filter_vecs)
    # print(np.reshape(kmeans.labels_, (8, 16)))
    return kmeans.labels_


def filters_clustering_quant(filters, n_clusters=0):
    filters_copy = copy.copy(filters)
    num_filters, num_layers, h, w = filters_copy.shape
    filter_vecs = np.zeros(shape=(num_filters, h*w*num_layers))
    for i in range(num_filters):
        filter_vecs[i] = filters_copy[i].reshape((1, h*w*num_layers))
    kmeans = KMeans(n_clusters=n_clusters, verbose=1).fit(filter_vecs)
    print(np.reshape(kmeans.labels_, (8, 16)))
    return kmeans.labels_


def filters_quant(filters):
    filters_data = copy.copy(filters.transpose((3, 2, 0, 1)))
    # min_val, max_val = filters_data.min(), filters_data.max()
    bins = np.arange(-1, 1, 0.005)
    filters_data_quant = np.digitize(filters_data, bins)
    return filters_data_quant

def sort_fmaps(feature_maps, method):
    """
    @feature_maps: original float feature maps of shape (None, 78, 78, 128)
    @method:
        lum: Sort feature maps by luminance
        random: random sort the feature maps
        glcm: collect glcm features and cluster them
    
    return: dl: sorted list
            fmaps: feature maps
    """
    fmaps = copy.copy(feature_maps)[0]
    fmaps = (fmaps - fmaps.min()) * 255 / (fmaps.max() - fmaps.min())
    fmaps = fmaps.astype(np.uint8).transpose((2, 0, 1))
    d = {}
    dl = []
    num_clusters = 5
    if method == 'lum':
        for idx, fmap in enumerate(fmaps):
            d[idx] = np.average(fmap)
        dl = sorted(d.items(), key=lambda x: x[1], reverse=False)
    if method == 'glcm':
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
    if method == 'random':
        for idx in range(128):
            d[idx] = idx
        dl = list(d.items())
        np.random.shuffle(dl)
    return dl, fmaps


# depreciated, using decode_result() instead
def decode(model_output, output_sizes=(19, 19), num_class=80, anchors=None):
    '''
     model_output: the feature of the output of yolo
     output_sizes: the size of output feature map
    '''
    H, W = output_sizes
    num_anchors = len(anchors)  # 这里的anchor是在configs文件中设置的
    anchors = tf.constant(anchors, dtype=tf.float32)  # 将传入的anchors转变成tf格式的常量列表

    # 13*13*num_anchors*(num_class+5)，第一个维度自适应batchsize
    detection_result = tf.reshape(
        model_output, [-1, H*W, num_anchors, num_class+5])

    # darknet19网络输出转化——偏移量、置信度、类别概率
    # 中心坐标相对于该cell左上角的偏移量，sigmoid函数归一化到0-1
    xy_offset = tf.nn.sigmoid(detection_result[:, :, :, 0:2])
    # 相对于anchor的wh比例，通过e指数解码
    wh_offset = tf.exp(detection_result[:, :, :, 2:4])
    obj_probs = tf.nn.sigmoid(
        detection_result[:, :, :, 4])  # 置信度，sigmoid函数归一化到0-1
    # 网络回归的是'得分',用softmax转变成类别概率
    class_probs = tf.nn.softmax(detection_result[:, :, :, 5:])

    # 构建特征图每个cell的左上角的xy坐标
    height_index = tf.range(H, dtype=tf.float32)  # range(0,13)
    width_index = tf.range(W, dtype=tf.float32)  # range(0,13)
    # 变成x_cell=[[0,1,...,12],...,[0,1,...,12]]和y_cell=[[0,0,...,0],[1,...,1]...,[12,...,12]]
    x_cell, y_cell = tf.meshgrid(height_index, width_index)
    # 和上面[H*W,num_anchors,num_class+5]对应
    x_cell = tf.reshape(x_cell, [1, -1, 1])
    y_cell = tf.reshape(y_cell, [1, -1, 1])

    # decode
    bbox_x = (x_cell + xy_offset[:, :, :, 0]) / W
    bbox_y = (y_cell + xy_offset[:, :, :, 1]) / H
    bbox_w = (anchors[:, 0] * wh_offset[:, :, :, 0]) / W
    bbox_h = (anchors[:, 1] * wh_offset[:, :, :, 1]) / H
    # 中心坐标+宽高box(x,y,w,h) -> xmin=x-w/2 -> 左上+右下box(xmin,ymin,xmax,ymax)
    bboxes = tf.stack([bbox_x-bbox_w/2, bbox_y-bbox_h/2,
                       bbox_x+bbox_w/2, bbox_y+bbox_h/2], axis=3)

    return bboxes, obj_probs, class_probs
