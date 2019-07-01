#!/bin/bash
base=/home/zrb/apps/build_env_vnf/container/yolov2
/home/zrb/tensorflow/bazel-bin/tensorflow/tools/benchmark/benchmark_model \
  --graph=${base}/model/part1_test.pb \
  --input_layer="input_1:0" \
  --input_layer_shape="1,608,608,3" \
  --input_layer_type="float" \
  --output_layer="max_pooling2d_3/MaxPool:0"