#!/bin/bash
base=/home/zrb/apps/build_env_vnf/container/yolov2
/home/zrb/tensorflow/bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
    --in_graph="${base}/model/part1_test.pb" \
    --out_graph="${base}/model/part1_test_quantized.pb" \
    --inputs="input_1" \
    --outputs="max_pooling2d_3/MaxPool" \
    --transforms="quantize_weights quantize_nodes"
