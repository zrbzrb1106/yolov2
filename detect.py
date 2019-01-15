import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np

from utils import *
from config import anchors, class_names


def read_model_pb_into_session(path):
    """
    args:
        path: model path in .pb format
    return:
        sess: tf.Session()
        layers_names: the name of layers including input and output layer
    """
    sess = tf.Session()
    with gfile.FastGFile(path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

    tensor_names = [t.name for op in sess.graph.get_operations()
                    for t in op.values()]
    layer_names = [name for name in tensor_names if (
        'conv' in name and 'filter' not in name) or 'pool' in name or 'input' in name or 'output' in name]
    print(layer_names)
    return sess, layer_names


def get_feature_map(orig, is_display=0, is_save=0):
    """
    orig: original feature map
    is_display: bool var for controlling display the feature map or not
    is_save: bool var for controlling save the feature map or not
    """
    feature_map = np.squeeze(orig, axis=0)
    m, n, c = feature_map.shape
    a = int(np.sqrt(c))
    while c % a is not 0:
        a = a - 1
    b = int(c / a)
    imgs = []
    cur = []

    for i in range(0, a):
        for j in range(0, b):
            tmp = np.zeros(shape=(orig.shape[0], orig.shape[1]))
            f_tmp = cv2.normalize(
                feature_map[:, :, i * j], tmp, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
            cur.append(f_tmp)
        f_tmp = np.hstack(cur)
        imgs.append(f_tmp)
        cur = []
    res = np.vstack(imgs)
    if is_display:
        cv2.imshow("feature", res)
        cv2.waitKey(0)
    if is_save:
        cv2.imwrite("./feature.jpg", res)
    return feature_map


def main():
    input_size = (608, 608)
    sess, layers = read_model_pb_into_session('./model/yolo.pb')
    img_orig = cv2.imread('./data/car.jpg')
    img = preprocess_image(img_orig)

    input_tensor = sess.graph.get_tensor_by_name(layers[0])
    output_tensor = sess.graph.get_tensor_by_name(layers[-1])
    # get feature map and if possible, display it
    # layers[num]: num is the num'th layer, e.g, layers[2] is the second layer, namely the first maxpool layer
    feature_map = sess.run(sess.graph.get_tensor_by_name(
        layers[5]), feed_dict={input_tensor: img})
    get_feature_map(feature_map, 0, 0)

    output_sizes = input_size[0]//32, input_size[1]//32
    output_decoded = decode(model_output=output_tensor, output_sizes=output_sizes,
                            num_class=len(class_names), anchors=anchors)
    bboxes, obj_probs, class_probs = sess.run(
        output_decoded, feed_dict={input_tensor: img})

    bboxes, scores, class_max_index = postprocess(
        bboxes, obj_probs, class_probs, image_shape=img_orig.shape[:2])

    img_detection = draw_detection(
        img_orig, bboxes, scores, class_max_index, class_names)
    cv2.imwrite("./yolo2_data/detection.jpg", img_detection)
    print('YOLO_v2 detection has done!')
    cv2.imshow("detection_results", img_detection)
    cv2.waitKey(0)
    cv2.imwrite("./data/detection.jpg", img_detection)


if __name__ == "__main__":
    main()
