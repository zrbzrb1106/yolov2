import tensorflow as tf
from tensorflow.python.client import timeline
from keras import backend as K
from keras.models import load_model

import numpy as np
import sys
sys.path.append('./')
from tools.model_proc import read_model

run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()

sess = read_model('./model/part1_test.pb', '')
# input_name = 'input:0'
# output_name = 'output:0'
input_name = 'input_1:0'
output_name = 'max_pooling2d_3/MaxPool:0'

input_tensor = sess.graph.get_tensor_by_name(input_name)
output_tensor = sess.graph.get_tensor_by_name(output_name)

for i in range(1):
    input_data = np.array(np.random.random_sample((1, 608, 608, 3)), dtype=np.float32)
    sess.run(output_tensor, feed_dict={input_tensor:input_data}, options=run_options, run_metadata=run_metadata)

    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    with open('./logs_pedes/timeline_%d.json' % (i), 'w') as f:
        f.write(ctf)
    options = tf.profiler.ProfileOptionBuilder.time_and_memory()
    options['min_bytes'] = 0
    options["min_micros"] = 0
    options['select'] = ("bytes", "peak_bytes", "output_bytes",
                         "residual_bytes")
    tf.profiler.profile(sess.graph, run_meta=run_metadata, options=options)
