import numpy as np
import matplotlib.pyplot as plt
from detect import *


def plot_two_parts_inference_time():
    N = 9
    inference_time_first = []
    inference_time_second = []
    inference_time_first_std = []
    inference_time_second_std = []
    batch_size = [1, 2, 4, 6, 8, 10, 12, 14, 16]
    batch_size = [1, 16, 32]
    sess, layers = read_model_pb_into_session('./model/yolo.pb')
    for b in batch_size:
        t1_mean, t2_mean, t1_std, t2_std = eval_model(b, sess, layers)
        print(t1_mean, t2_mean, t1_std, t2_std)
        inference_time_first.append(t1_mean)
        inference_time_second.append(t2_mean)
        inference_time_first_std.append(t1_std)
        inference_time_second_std.append(t2_std)

    ind = np.arange(N)    # the x locations for the groups

    p1 = plt.bar(ind, inference_time_first, width=0.5,
            yerr=inference_time_first_std)
    p2 = plt.bar(ind, inference_time_second, width=0.5,
            bottom=inference_time_first, yerr=inference_time_second_std)

    plt.ylabel('Seconds per frame')
    plt.title('Inference time by different batch size')
    plt.xticks(ind, ('1', '2', '4', '6', '8', '10', '12', '14', '16'))
    plt.yticks(np.arange(0, 1.5, 0.1))
    plt.legend((p1[0], p2[0]), ('first_part', 'second_part'))


    plt.show()
    # plt.savefig("dev/plots/inference_batch_time.png")

plot_two_parts_inference_time()