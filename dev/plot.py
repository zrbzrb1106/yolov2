import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from scipy.stats import t
from scipy import stats
import sys
import matplotlib
sys.path.append('./')
from tools import tex

def calc_ci(data):
    n = len(data)
    data_mean = np.mean(data)
    data_std = np.std(data)
    conf_interval = stats.norm.interval(0.95, data_mean, data_std)
    return data_mean, data_mean-conf_interval[0]


def parse_logfile(file_path, idx, is_skip_firstline=False):
    f = open(file_path)
    if is_skip_firstline:
        data = f.readlines()[1:]
    else:
        data = f.readlines()
    res = [int(i.split()[idx]) for i in data]
    return res


def plot_fmaps_space_analysis():
    tex.setup(width=1, l=0.25, r=0.95, b=0.1, t=0.9)
    f = open('./logs_pedes/log_space_analysis_coco.log')
    data = f.readlines()
    res = np.zeros(shape=(5000, 6))
    for ln, line in enumerate(data):
        counts = np.array([int(i) for i in line.split()])
        res[ln] = counts
    ave = np.average(res, 0)
    total = ave[0]
    r = []
    last = 0
    for i in ave[1:]:
        cur = i - last
        tmp = cur / total * 100
        last = i
        r.append(tmp)
    r.append((total - ave[-1]) / total * 100)
    x = np.arange(6)
    xnames = ['[0, R/100)', '[R/100, R/50)', '[R/50, R/20)', '[R/20, R/10)', '[R/10, R/5)', 'Others']
    fig, ax = plt.subplots()
    ax.spines['right'].set_position(('data', 51))
    ax.spines['top'].set_bounds(0, 51)
    ax.spines['bottom'].set_bounds(0, 51)
    ax.barh(x, r, color='b', align='center')
    for a, b in zip(x, r):
        ax.text(b, a, str(np.round(b, 1)))
    ax.set_yticks(x)
    ax.set_yticklabels(xnames)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Percentage of parameters')
    ax.set_xticks([], [])
    plt.grid()
    plt.savefig('C:\\d\\diplomarbeit\\DA\\Figures\\fmaps_space_analysis.pdf',
                format='pdf', dpi=1000)
    plt.show()
    


def accu_ratio(method='jpeg'):
    '''
    method = 'jpeg' or 'webp'
    '''
    tex.setup(width=2, l=0.1, r=0.9, b=0.1, t=0.9)
    x = np.arange(20, 100, 10)
    baseline_raw = 608 * 608 * 3
    baseline_jpg = parse_logfile('./logs_pedes/resized_size.log', 0)

    accu_baseline = 23.3
    jpg_accu = [13.9, 18.8, 21.0, 22.1, 22.6, 23.0, 23.2, 23.2]
    jpg_accu_delta = np.array(jpg_accu) / accu_baseline
    webp_accu = [19.0, 21.4, 22.4, 22.8, 23.0, 23.0, 23.2, 23.3]
    webp_accu_delta = np.array(webp_accu) / accu_baseline

    if method == 'jpeg':
        base = './logs_pedes/log_jpeg_q_{}.log'
        accu = jpg_accu_delta * 100
    if method == 'webp':
        base = './logs_pedes/log_webp_q_{}.log'
        accu = webp_accu_delta * 100

    qs = np.arange(20, 100, 10)
    ratios_raw = []
    ratios_jpg = []
    err_raw = []
    err_jpg = []
    for q in qs:
        data = parse_logfile(base.format(q), 2, False)
        data_raw = np.array(data) / baseline_raw * 100
        data_jpg = np.array(data) / baseline_jpg * 100
        raw_mean, raw_err = calc_ci(data_raw)
        jpg_mean, jpg_err = calc_ci(data_jpg)
        ratios_raw.append(raw_mean)
        ratios_jpg.append(jpg_mean)
        err_raw.append(raw_err)
        err_jpg.append(jpg_err)

    color1 = 'darkblue'
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Quality Factor')
    ax1.set_ylabel('Average Precision(\%)', color=color1)
    ax1.plot(x, accu, color=color1)
    ax1.scatter(x, accu, color=color1)
    for a, b in zip(x, np.round(accu, 1)):
        ax1.text(a - 3, b + 1, str(b), color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    plt.ylim(50, 103)

    ax2 = ax1.twinx()
    ax2.set_ylabel(
        'Size of compressed image/Size of original image(\%)', color='k')
    ax2.bar(x - 0.75, ratios_jpg, width=1.5, yerr=err_jpg,
            align='center', color='#f33c74', hatch='//')
    for a, b in zip(x, np.round(ratios_jpg, 1)):
        ax2.text(a-0.5, b + 0.5, str(b))
    ax2.bar(x + 0.75, ratios_raw, width=1.5, yerr=err_raw,
            align='center', color='purple', hatch='x')
    for a, b in zip(x, np.round(ratios_raw, 1)):
        ax2.text(a + 1, b + 1, str(b))
    ax2.tick_params(axis='y', labelcolor='k')
    ax2.legend(('JPEG Input', 'Raw Input'), loc=2)
    # plt.ylim(0, 130)
    plt.xticks(x)
    plt.savefig('C:\\d\\diplomarbeit\\DA\\Figures\\ratio_ap_quality_{}.pdf'.format(method), format='pdf', dpi=1000)
    plt.show()


def plot_ratio_quality():

    x = np.arange(20, 100, 10)
    # jpg_ratio = [7.70, 5.21, 4.07, 3.35, 2.80, 2.23, 1.68, 1.10]
    # webp_ratio = [7.01, 5.04, 3.97, 3.30, 2.83, 2.44, 1.86, 1.20]
    tex.setup(width=1, l=0.15, r=0.9, b=0.15, t=0.9)
    def get_plot_data(method):
        baseline_raw = 608 * 608 * 3
        baseline_jpg = parse_logfile('./logs_pedes/resized_size.log', 0)
        if method == 'jpeg':
            base = './logs_pedes/log_jpeg_q_{}.log'
        if method == 'webp':
            base = './logs_pedes/log_webp_q_{}.log'

        qs = np.arange(20, 100, 10)
        ratios_raw = []
        ratios_jpg = []
        err_raw = []
        err_jpg = []
        for q in qs:
            # if webp, then False
            data = parse_logfile(base.format(q), 2, False)
            data_raw = np.array(data) / baseline_raw * 100
            data_jpg = np.array(data) / baseline_jpg * 100
            raw_mean, raw_err = calc_ci(data_raw)
            jpg_mean, jpg_err = calc_ci(data_jpg)
            ratios_raw.append(raw_mean)
            ratios_jpg.append(jpg_mean)
            err_raw.append(raw_err)
            err_jpg.append(jpg_err)
        return ratios_raw, ratios_jpg, err_raw, err_jpg

    ratios_raw_jpeg, ratios_jpg_jpeg, err_raw_jpeg, err_jpg_jpeg = get_plot_data(
        'jpeg')
    ratios_raw_webp, ratios_jpg_webp, err_raw_webp, err_jpg_webp = get_plot_data(
        'webp')

    fig = plt.figure()

    plt.errorbar(x+0.25, ratios_raw_jpeg,
                 yerr=err_raw_jpeg, color='b', capsize=4)
    plt.errorbar(x-0.25, ratios_raw_webp, yerr=err_raw_webp,
                 fmt='--', color='darkblue', capsize=4)

    plt.errorbar(x+0.25, ratios_jpg_jpeg,
                 yerr=err_jpg_jpeg, color='g', capsize=4)
    plt.errorbar(x-0.25, ratios_jpg_webp, yerr=err_jpg_webp,
                 color='darkgreen', fmt='--', capsize=4)

    plt.xlabel('Quality')
    plt.ylabel('Compressed size/Original size(\%)')
    plt.legend(('JPEG of Raw Input.', 'WEBP of Raw Input.',
                'JPEG of JPEG Input.', 'WEBP of JPEG Input.'))
    plt.savefig('C:\\d\\diplomarbeit\\DA\\Figures\\ratio_quality_comp.pdf',
                format='pdf', dpi=1000)
    plt.show()


def plot_accu_quality():
    tex.setup(width=1, l=0.15, r=0.9, b=0.15, t=0.9)
    x = np.arange(20, 100, 10)
    jpg_accu = np.array([13.9, 18.8, 21.0, 22.1, 22.6,
                         23.0, 23.2, 23.2]) / 23.3 * 100
    webp_accu = np.array([19.0, 21.4, 22.4, 22.8, 23.0,
                          23.0, 23.2, 23.3]) / 23.3 * 100

    fig = plt.figure()

    plt.plot(x, jpg_accu, 'g', ls='--')
    plt.scatter(x, jpg_accu, color='', marker='o', edgecolors='g')
    plt.plot(x, webp_accu, 'b')
    plt.scatter(x, webp_accu, color='', marker='o', edgecolors='b')
    plt.xlabel('Quality')
    plt.ylabel('Average Precision(\%)')
    plt.legend(('JPEG', 'WEBP'))

    plt.savefig('C:\\d\\diplomarbeit\\DA\\Figures\\accu_quality_comp.pdf',
                format='pdf', dpi=1000)
    plt.show()


def plot_intra_ratio_ap():
    tex.setup(width=2, l=0.1, r=0.9, b=0.1, t=0.9)
    def get_plot_data(method):
        baseline_raw = 608 * 608 * 3
        baseline_jpg = parse_logfile('./logs_pedes/resized_size.log', 0)
        if method == 'jpeg':
            base = './logs_pedes/log_jpeg_q_{}.log'
        if method == 'webp':
            base = './logs_pedes/log_webp_q_{}.log'

        qs = np.arange(20, 100, 10)
        ratios_raw = []
        ratios_jpg = []
        err_raw = []
        err_jpg = []
        for q in qs:
            # if webp, then False
            data = parse_logfile(base.format(q), 2, False)
            data_raw = np.array(data) / baseline_raw * 100
            data_jpg = np.array(data) / baseline_jpg * 100
            raw_mean, raw_err = calc_ci(data_raw)
            jpg_mean, jpg_err = calc_ci(data_jpg)
            ratios_raw.append(raw_mean)
            ratios_jpg.append(jpg_mean)
            err_raw.append(raw_err)
            err_jpg.append(jpg_err)
        return ratios_raw, ratios_jpg, err_raw, err_jpg
    ratios_raw_jpeg, ratios_jpg_jpeg, err_raw_jpeg, err_jpg_jpeg = get_plot_data(
        'jpeg')
    ratios_raw_webp, ratios_jpg_webp, err_raw_webp, err_jpg_webp = get_plot_data(
        'webp')
    # jpg_ratio = [7.70, 5.21, 4.07, 3.35, 2.80, 2.23, 1.68, 1.10]
    jpg_accu = np.array([13.9, 18.8, 21.0, 22.1, 22.6,
                         23.0, 23.2, 23.2]) / 23.3 * 100

    # webp_ratio = [7.01, 5.04, 3.97, 3.30, 2.83, 2.44, 1.86, 1.20]
    webp_accu = np.array([19.0, 21.4, 22.4, 22.8, 23.0,
                          23.0, 23.2, 23.3]) / 23.3 * 100
    # x = np.arange(13, 25, 1)
    plt.figure()
    plt.errorbar(jpg_accu, ratios_raw_jpeg,
                 yerr=err_raw_jpeg, color='b', capsize=4)
    plt.errorbar(webp_accu, ratios_raw_webp, yerr=err_raw_webp,
                 color='darkblue', fmt='--', capsize=4)
    plt.errorbar(jpg_accu, ratios_jpg_jpeg,
                 yerr=err_jpg_jpeg, color='g', capsize=4)
    plt.errorbar(webp_accu, ratios_jpg_webp, yerr=err_jpg_webp,
                 color='darkgreen', fmt='--', capsize=4)
    # plt.xticks(x)
    # plt.scatter(jpg_accu, ratios_raw_jpeg, marker='o', linewidths=1)
    # plt.scatter(webp_accu, webp_ratio, marker='v', linewidths=1)
    plt.xlabel('Average Precision(\%)')
    plt.ylabel('Size of compressed image/Size of original image(\%)')
    plt.legend(('JPEG of Raw Input.', 'WEBP of Raw Input.',
                'JPEG of JPEG Input.', 'WEBP of JPEG Input.'))
    plt.savefig('C:\\d\\diplomarbeit\\DA\\Figures\\intra_ratio_accu.pdf',
                format='pdf', dpi=1000)
    plt.show()


def plot_compressibility_approaches():
    tex.setup(width=2, l=0.1, r=0.9, t=0.9, b=0.1)
    x = np.arange(7)
    xnames = ['Baseline', 'zlib', '3 decimals', '2 decimals',
              '1 decimal', 'Normed Cut', 'Cut']
    patterns = ['/', '\\', '+']
    base = './logs_pedes/'
    files = ['log_float_decimals_10.log', 'log_float_decimals_3.log', 'log_float_decimals_2.log',
             'log_float_decimals_1.log', 'log_cut_norm.log', 'log_cut_orig.log']
    size_orig = 3115008
    ratio = [1]
    ratio_err = [0]
    for f in files:
        f_name = base + f
        data = open(f_name).readlines()[1:]
        ratios = size_orig / np.array([int(i.split()[2]) for i in data])
        mean, err = calc_ci(ratios)
        ratio.append(mean)
        ratio_err.append(err)

    ap = np.array([23.3, 23.3, 23.3, 23.3, 23.3, 23.3, 22.9]) / 23.3 * 100

    fig, ax1 = plt.subplots()
    color1 = 'darkblue'
    ax1.set_ylabel('Average Precision(\%)', color=color1)
    l1 = ax1.scatter(x - 0.2, ap,color=color1)
    for a, b in zip(x, np.round(ap, 1)):
        ax1.text(a - 0.4, b + 0.3, str(b), color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    plt.ylim(95, 102)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Compressibility(times)', color='k')
    l2 = ax2.bar(x, ratio, yerr=ratio_err, width=0.2, align='center',
                 color='lightslategrey', hatch='//')
    for a, b in zip(x, np.round(ratio, 1)):
        ax2.text(a, b + 0.4, str(b)+'x')
    ax2.tick_params(axis='y', labelcolor='k')
    plt.xticks(x, xnames)
    ax2.legend([l1, l2], ('Average Precision', 'Compression Ratio'))
    plt.savefig('C:\\d\\diplomarbeit\\DA\\Figures\\approaches.pdf',
                format='pdf', dpi=1000)
    plt.show()


'''
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
'''
# new images


def intra_speed_ratio():
    def parse_log(file, method):
        ref_size = open('./logs/resized_size.log').readlines()
        f = open(file)
        data = f.readlines()
        if method == 'jpeg':
            data = data[1:]
        if method == 'webp':
            pass
        ratios = []
        cnt = 0
        t_enc = []
        t_dec = []
        for line in data:
            t1, t2, s = line.split()
            ratio = float(ref_size[cnt]) / int(s)
            t_enc.append(float(t1))
            t_dec.append(float(t2))
            ratios.append(ratio)
            cnt += 1
        ratio = sum(ratios) / len(ratios)
        t_enc_ave = sum(t_enc) / len(t_enc)
        t_dec_ave = sum(t_dec) / len(t_dec)
        return t_enc_ave, t_dec_ave, ratio
    methods = ['jpeg', 'webp']
    d = {}
    qs = range(20, 100, 10)
    base = './logs/'
    for method in methods:
        speeds_enc = []
        speeds_dec = []
        ratios = []
        for q in qs:
            filename = 'log_{}_q_{}.log'.format(method, q)
            speed_1, speed_2, ratio = parse_log(base + filename, method)
            speeds_enc.append(speed_1)
            speeds_dec.append(speed_2)
            ratios.append(ratio)
        d[method] = (speeds_enc, speeds_dec, ratios)
    # plot
    jpg_accu = [13.9, 18.8, 21.0, 22.1, 22.6, 23.0, 23.2, 23.2]
    webp_accu = [19.0, 21.4, 22.4, 22.8, 23.0, 23.0, 23.2, 23.3]
    x = d['jpeg'][2]
    plt.figure(1)
    plt.subplot(221)
    plt.plot(d['jpeg'][0], x, '-')
    plt.plot(d['webp'][0], x)
    plt.ylabel('Compression Ratio (times)')
    plt.xlabel('Encoding Time (Seconds)')
    plt.legend(('JPEG', 'WEBP'))
    plt.title('Encoding Time')
    plt.xlim(0, 0.175)
    plt.subplot(222)
    plt.plot(d['jpeg'][1], x, '-')
    plt.plot(d['webp'][1], x)
    plt.ylabel('Compression Ratio (times)')
    plt.xlabel('Decoding Time (Seconds)')
    plt.legend(('JPEG', 'WEBP'))
    plt.title('Decoding Time')
    plt.xlim(0, 0.04)
    plt.subplot(223)
    plt.scatter(d['jpeg'][0], jpg_accu, marker='v')
    plt.scatter(d['webp'][0], webp_accu)
    plt.ylabel('Average Precision')
    plt.xlabel('Encoding Time (Seconds)')
    plt.legend(('JPEG', 'WEBP'))
    plt.xlim(0, 0.175)
    plt.subplot(224)
    plt.scatter(d['jpeg'][1], jpg_accu, marker='v')
    plt.scatter(d['webp'][1], webp_accu)
    plt.ylabel('Average Precision')
    plt.xlabel('Decoding Time (Seconds)')
    plt.legend(('JPEG', 'WEBP'))
    plt.xlim(0, 0.04)
    # plt.suptitle('Performance Comparision of JPEG and WEBP')
    plt.show()


def plot_prune_ap():
    tex.setup(width=2, l=0.1, r=0.9, b=0.1, t=0.9)
    x = np.array([10, 20, 40, 60, 80, 100, 120, 140, 160, 180])
    accu = np.array([20.0, 22.8, 23.1, 23.1, 23.2, 23.2,
                     23.3, 23.3, 23.3, 23.3]) / 23.3 * 100
    base = './logs_pedes/log_prune_factor_{}.log'
    pruned = []
    errs = []
    for pf in x:
        data = parse_logfile(base.format(pf), 3, False)
        ratio = (778752 - np.array(data)) / 778752 * 100
        mean, err = calc_ci(ratio)
        pruned.append(mean)
        errs.append(err)

    color1 = 'darkblue'
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Prune Factor')
    ax1.set_ylabel('Average Precision(\%)', color=color1)
    ax1.plot(x, accu, color=color1)
    ax1.scatter(x, accu, color=color1)
    for a, b in zip(x, np.round(accu, 1)):
        ax1.text(a - 5, b+0.8, str(b), color=color1)
    ax1.tick_params(axis='y', labelcolor='k')
    plt.ylim(70, 102)

    ax2 = ax1.twinx()
    ax2.set_ylabel(
        'Params after pruning/Params before pruning(\%)', color='k')
    ax2.bar(x, pruned, width=5, yerr=errs,
            align='center', color='lightslategrey', hatch='//')
    for a, b in zip(x, np.round(pruned, 1)):
        ax2.text(a, b + 5, str(b))
    ax2.tick_params(axis='y', labelcolor='k')
    plt.xticks(x)
    plt.ylim(0, 100)
    plt.grid()
    plt.savefig('C:\\d\\diplomarbeit\\DA\\Figures\\prune_performance.pdf',
                format='pdf', dpi=1000)
    plt.show()


def plot_image_based_all(base='jpg'):
    """
    base: 'jpg' for jpeg input
          'raw' for raw input
    """
    tex.setup(width=2, l=0.1, r=0.9, b=0.1, t=0.9)
    baseline_raw = 608 * 608 * 3
    baseline_jpg = parse_logfile('./logs_pedes/resized_size.log', 0)

    jpg_accu = np.array([13.9, 18.8, 21.0, 22.1, 22.6,
                         23.0, 23.2, 23.2]) / 23.3 * 100
    webp_accu = np.array([19.0, 21.4, 22.4, 22.8, 23.0,
                          23.0, 23.2, 23.3]) / 23.3 * 100
    h264_slower_accu = np.array(
        [2.1, 8.9, 16.3, 19.9, 21.1, 21.5]) / 23.3 * 100
    h264_medium_accu = np.array(
        [1.4, 7.0, 14.5, 19.1, 20.8, 21.4]) / 23.3 * 100
    h264_faster_accu = np.array(
        [0.9, 4.8, 12.8, 18.1, 20.3, 21.4]) / 23.3 * 100
    h264_ultrafast_accu = np.array(
        [15, 18.7, 20.2, 20.9, 21.3, 21.4]) / 23.3 * 100

    def get_plot_data(method):
        if method == 'jpeg':
            base = './logs_pedes/log_jpeg_q_{}.log'
        if method == 'webp':
            base = './logs_pedes/log_webp_q_{}.log'
        if method == 'h264_slower':
            base = './logs_pedes/log_h264_preset_slower_crf_{}.log'
        if method == 'h264_medium':
            base = './logs_pedes/log_h264_preset_medium_crf_{}.log'
        if method == 'h264_faster':
            base = './logs_pedes/log_h264_preset_faster_crf_{}.log'
        if method == 'h264_ultrafast':
            base = './logs_pedes/log_h264_preset_ultrafast_crf_{}.log'

        qs = np.arange(20, 100, 10) # 20, 30, 40, ..., 90
        crfs = np.arange(30, 18, -2) # 30 28, 26, 24, 22, 20
        params = [qs, crfs]
        flag = 1
        if 'jpeg' in method or 'webp' in method:
            flag = 0
        ratios_raw = []
        ratios_jpg = []
        err_raw = []
        err_jpg = []
        for param in params[flag]:
            data = parse_logfile(base.format(param), 2, False)
            data_raw = np.array(data) / baseline_raw * 100
            data_jpg = np.array(data) / baseline_jpg * 100
            raw_mean, raw_err = calc_ci(data_raw)
            jpg_mean, jpg_err = calc_ci(data_jpg)
            ratios_raw.append(raw_mean)
            ratios_jpg.append(jpg_mean)
            err_raw.append(raw_err)
            err_jpg.append(jpg_err)
        return ratios_raw, ratios_jpg, err_raw, err_jpg

    ratios_raw_jpeg, ratios_jpg_jpeg, err_raw_jpeg, err_jpg_jpeg = get_plot_data(
        'jpeg')
    ratios_raw_webp, ratios_jpg_webp, err_raw_webp, err_jpg_webp = get_plot_data(
        'webp')
    ratios_raw_h264_slower, ratios_jpg_h264_slower, err_raw_h264_slower, err_jpg_h264_slower = get_plot_data(
        'h264_slower')
    ratios_raw_h264_medium, ratios_jpg_h264_medium, err_raw_h264_medium, err_jpg_h264_medium = get_plot_data(
        'h264_medium')
    ratios_raw_h264_faster, ratios_jpg_h264_faster, err_raw_h264_faster, err_jpg_h264_faster = get_plot_data(
        'h264_faster')
    ratios_raw_h264_ultrafast, ratios_jpg_h264_ultrafast, err_raw_h264_ultrafast, err_jpg_h264_ultrafast = get_plot_data(
        'h264_ultrafast')

    fig = plt.figure()
    ratios = {'jpg': [ratios_jpg_jpeg, ratios_jpg_webp, ratios_jpg_h264_slower, 
                    ratios_jpg_h264_medium, ratios_jpg_h264_faster, ratios_jpg_h264_ultrafast],
              'raw': [ratios_raw_jpeg, ratios_raw_webp, ratios_raw_h264_slower, 
                    ratios_raw_h264_medium, ratios_raw_h264_faster, ratios_raw_h264_ultrafast]}

    errs = {'jpg': [err_jpg_jpeg, err_jpg_webp, err_jpg_h264_slower, 
                err_jpg_h264_medium, err_jpg_h264_faster, err_jpg_h264_ultrafast], 
            'raw': [err_raw_jpeg, err_raw_webp, err_raw_h264_slower,
                err_raw_h264_medium, err_raw_h264_faster, err_raw_h264_ultrafast]}

    plt.errorbar(jpg_accu, ratios[base][0], yerr=errs[base][0], 
                 color='g', capsize=2)
    plt.errorbar(webp_accu, ratios[base][1], yerr=errs[base][1],
                 color='m', capsize=2)
    plt.errorbar(h264_slower_accu, ratios[base][2], yerr=errs[base][2],
                 color='b', capsize=2)
    plt.errorbar(h264_medium_accu, ratios[base][3], yerr=errs[base][3],
                 color='r', capsize=2)
    plt.errorbar(h264_faster_accu, ratios[base][4], yerr=errs[base][4],
                 color='k', capsize=2)
    plt.errorbar(h264_ultrafast_accu, ratios[base][5], yerr=errs[base][5],
                 color='y', capsize=2)
    plt.xlabel('Average Precision(\%)')
    plt.ylabel('Size of compressed image/Size of original image(\%)')
    # plt.legend(('H264 slower', 'H264 medium', 'H264 faster', 'H264 ultrafast'))
    plt.legend(('JPEG', 'WEBP', 'H264 slower', 'H264 medium', 'H264 faster', 'H264 ultrafast'))
    plt.xlim(80, 102)
    plt.xticks(np.arange(80, 102, step=1))
    if base == 'jpg':
        ax1 = plt.gca()
        print(ax1.get_xlim())
        plt.axvline(98.7, 0, 50 / ax1.get_ylim()[1], linestyle='--')
        plt.axhline(50, 0, (98.7 - 80) / (ax1.get_xlim()[1] - 80), linestyle='--')

    left, bottom, width, height = 0.3, 0.5, 0.2, 0.3
    ax2 = fig.add_axes([left, bottom, width, height])
    ax2.errorbar(jpg_accu, ratios[base][0], yerr=errs[base][0], 
                 color='g', capsize=2)
    ax2.errorbar(webp_accu, ratios[base][1], yerr=errs[base][1],
                 color='m', capsize=2)
    ax2.errorbar(h264_slower_accu, ratios[base][2], yerr=errs[base][2],
                 color='b', capsize=2)
    ax2.errorbar(h264_medium_accu, ratios[base][3], yerr=errs[base][3],
                 color='r', capsize=2)
    ax2.errorbar(h264_faster_accu, ratios[base][4], yerr=errs[base][4],
                 color='k', capsize=2)
    ax2.errorbar(h264_ultrafast_accu, ratios[base][5], yerr=errs[base][5],
                 color='y', capsize=2)
    ax2.set_xlim(80, 90)
    if base == 'jpg':
        ax2.set_ylim(15, 50)
    else:
        ax2.set_ylim(2.5, 12.5)

    # left, bottom, width, height = 0.5, 0.4, 0.2, 0.3
    # ax3 = fig.add_axes([left, bottom, width, height])
    # ax3.errorbar(jpg_accu, ratios[base][0], yerr=errs[base][0], 
    #              color='g', capsize=2)
    # ax3.errorbar(webp_accu, ratios[base][1], yerr=errs[base][1],
    #              color='m', capsize=2)
    # ax3.errorbar(h264_slower_accu, ratios[base][2], yerr=errs[base][2],
    #              color='b', capsize=2)
    # ax3.errorbar(h264_medium_accu, ratios[base][3], yerr=errs[base][3],
    #              color='r', capsize=2)
    # ax3.errorbar(h264_faster_accu, ratios[base][4], yerr=errs[base][4],
    #              color='k', capsize=2)
    # ax3.errorbar(h264_ultrafast_accu, ratios[base][5], yerr=errs[base][5],
    #              color='y', capsize=2)
    # ax3.set_xlim(90, 101)

    # plt.savefig('C:\\d\\diplomarbeit\\DA\\Figures\\h264_comp_jpg.pdf'.format(base),
    #             format='pdf', dpi=1000)
    plt.savefig('C:\\d\\diplomarbeit\\DA\\Figures\\image_based_approached_{}.pdf'.format(base),
                format='pdf', dpi=1000)
    plt.show()


def plot_data_based_all(input_data='jpg'):
    """
    base: 'jpg' for JPEG Input
          'raw' for Raw Input
    """
    tex.setup(width=2, l=0.1, r=0.9, b=0.1, t=0.9)
    baseline_raw = 608 * 608 * 3
    baseline_jpg = parse_logfile('./logs_pedes/resized_size.log', 0)
    ap = np.array([23.3, 23.3, 23.3, 22.9, 22.9]) / 23.3 * 100
    xnames = ['zlib', '1 decimal','NormedCut+zlib', 'Cut+zlib', 'Cut+PNG']
    x = np.arange(5)
    base = './logs_pedes/'
    files = ['log_float_decimals_10.log', 'log_float_decimals_1.log', 
        'log_cut_norm.log','log_cut_orig.log', 'log_png.log']
    baseline = [baseline_jpg, baseline_raw]
    flag = 0 if input_data == 'jpg' else 1
    ratio = []
    ratio_err = []
    for f in files:
        f_name = base + f
        data = open(f_name).readlines()
        ratios = np.array([int(i.split()[2]) for i in data]) / baseline[flag] * 100
        mean, err = calc_ci(ratios)
        ratio.append(mean)
        ratio_err.append(err)

    fig, ax1 = plt.subplots()
    color1 = 'darkblue'
    ax1.set_xlabel('Approaches for Feature Maps Compressibility Evaluation')
    ax1.set_ylabel('Average Precision(\%)', color=color1)
    l1 = ax1.scatter(x , ap, color=color1)
    ax1.plot(x, ap, color=color1)
    for a, b in zip(x, np.round(ap, 1)):
        ax1.text(a, b + 0.2, str(b), color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    plt.ylim(90, 102)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Compressed size/Original size(\%)', color='k')
    l2 = ax2.bar(x, ratio, yerr=ratio_err, width=0.2, capsize=2, align='center',
                 color='#f33c74', hatch='x')
    for a, b in zip(x, np.round(ratio, 1)):
        ax2.text(a + 0.05, b + 4, str(b))
    ax2.tick_params(axis='y', labelcolor='k')
    if input_data == 'raw':
        plt.ylim(0, 200)
    else:
        pass
    plt.xticks(x, xnames)
    ax2.legend([l1, l2], ('Average Precision', 'Compression Performance'))
    plt.savefig('C:\\d\\diplomarbeit\\DA\\Figures\\data_based_approached_{}.pdf'.format(input_data),
                format='pdf', dpi=1000)
    plt.show()


def plot_sort_h264():
    tex.setup(width=2, l=0.1, r=0.9, b=0.1, t=0.9)
    methods = ['lum', 'glcm', 'random']
    crfs = [25, 20]
    x = np.arange(3)
    accu_lum = np.array([17.2, 21.4]) / 23.3 * 100
    accu_glcm = np.array([17.4, 21.4]) / 23.3 * 100
    accu_random = np.array([17.3, 21.5]) / 23.3 * 100
    baseline_jpg = parse_logfile('./logs_pedes/resized_size.log', 0)
    base = './logs_pedes/log_h264_sort_{}_crf_{}.log'
    ratios = {'lum':[], 'glcm':[], 'random':[]}
    ratios_err = {'lum':[], 'glcm':[], 'random':[]}
    for method in methods:
        for crf in crfs:
            data = parse_logfile(base.format(method, crf), 2, False)
            data_jpg = np.array(data) / baseline_jpg * 100
            jpg_mean, jpg_err = calc_ci(data_jpg)
            ratios[method].append(jpg_mean)
            ratios_err[method].append(jpg_err)

    fig = plt.figure()
    plt.errorbar(accu_lum, ratios['lum'], yerr=ratios_err['lum'], 
                 color='g', capsize=2)
    plt.errorbar(accu_glcm, ratios['glcm'], yerr=ratios_err['glcm'], 
                color='b', capsize=2)
    plt.errorbar(accu_random, ratios['random'], yerr=ratios_err['random'], 
                 color='r', capsize=2)
    plt.legend(('lum', 'glcm', 'random'))
    plt.xlabel('Average precision(\%)')
    plt.ylabel('Size of compressed feature maps/Size of original feature maps(\%)')
    plt.savefig('C:\\d\\diplomarbeit\\DA\\Figures\\h264_sort_methods.pdf',
                format='pdf', dpi=1000)
    plt.show()


def plot_png_zlib():
    tex.setup(width=2, l=0.1, r=0.9, b=0.1, t=0.9)
    zlib_data_path = './logs_pedes/log_cut_orig.log'
    png_data_path = './logs_pedes/log_png.log'
    baseline_jpg = parse_logfile('./logs_pedes/resized_size.log', 0)
    baseline_raw = 608 * 608 * 3

    x = np.arange(2)
    x_names = ['PNG', 'zlib']

    png_size = parse_logfile(png_data_path, 2)
    zlib_size = parse_logfile(zlib_data_path, 2)
    files = [png_data_path, zlib_data_path]

    ratios_jpg = []
    ratios_raw = []
    errs_jpg = []
    errs_raw = []

    for i in range(2):
        size = parse_logfile(files[i], 2)
        tmp_jpg = np.array(size) / baseline_jpg * 100
        tmp_raw = np.array(size) / baseline_raw * 100
        ratio_jpg, err_jpg = calc_ci(tmp_jpg)
        ratio_raw, err_raw = calc_ci(tmp_raw)
        ratios_jpg.append(ratio_jpg)
        ratios_raw.append(ratio_raw)
        errs_jpg.append(err_jpg)
        errs_raw.append(err_raw)
    
    fig = plt.figure()
    plt.bar(x+0.05, ratios_jpg, yerr=errs_jpg, width=0.1, align='center',
                 color='#f33c74', hatch='x')
    for a, b in zip(x, np.round(ratios_jpg, 1)):
        plt.text(a-0.01, b + 2, str(b) + ' / 98.3%', color='k')
    plt.bar(x-0.05, ratios_raw, yerr=errs_raw, width=0.1, align='center',
                 color='b', hatch='//')
    for a, b in zip(x, np.round(ratios_raw, 1)):
        plt.text(a - 0.15, b + 2, str(b) + ' / 98.3%', color='k')
    plt.legend(('JPEG Input', 'Raw Input'), loc='upper left')
    plt.ylabel('Size of compressed feature maps/Size of original feature maps(\%)')
    plt.xlabel('Compression methods')
    plt.xticks(x, x_names)
    plt.savefig('C:\\d\\diplomarbeit\\DA\\Figures\\png_zlib.pdf',
                format='pdf', dpi=1000)
    plt.show()


def test():
    baseline_raw = 608 * 608 * 3
    baseline_jpg = parse_logfile('./logs_pedes/resized_size.log', 0)
    data_no_prune = parse_logfile('./logs_pedes/log_float_decimals_1.log', 2)
    data = parse_logfile('./logs_pedes/log_prune_factor_40_1_decimal.log', 2)

    a1 = np.array(data) / baseline_jpg * 100
    a2 = np.array(data) / baseline_raw * 100

    b1 = np.array(data_no_prune) / baseline_jpg * 100
    b2 = np.array(data_no_prune) / baseline_raw * 100

    ratio_jpg, err_jpg = calc_ci(a1)
    ratio_raw, err_raw = calc_ci(a2)

    ratio_jpg_no_prune, _ = calc_ci(b1)
    ratio_raw_no_prune, _ = calc_ci(b2)
    print(ratio_jpg, ratio_raw, ratio_jpg_no_prune, ratio_raw_no_prune)

def test_batched_h264():
    data = parse_logfile('./logs_pedes/log_batched_h264_crf_20_batchsize_10.log', 4)
    data_orig = parse_logfile('./logs_pedes/log_h264_preset_medium_crf_20.log', 2)
    mean_orig, _ = calc_ci(data_orig)
    mean_batched, _ = calc_ci(data)
    print(mean_orig, mean_batched // 10 * 128)


def plot_params_summary():
    tex.setup(width=2, l=0.1, r=0.9, b=0.3, t=0.7)
    filepath = './logs_pedes/params.txt'
    with open(filepath, 'r') as f:
        content = f.readlines()[1:]
    outputs = []
    res_d = {}
    for line in content:
        data = line.split()
        layer_num = int(data[0])
        name = data[1]+'\_'+'{}'.format(layer_num+1)
        if data[1] == 'conv':
            res_d[name] = np.prod([int(i) for i in data[5].split('x')])
        elif data[1] == 'max' or data[1]=='reorg':
            res_d[name] = np.prod([int(i) for i in data[4].split('x')])
        else:
            continue
    res_d = sorted(res_d.items(), key=lambda x: int(x[0].split('\_')[1]))
    fig= plt.figure()
    ax = plt.gca()
    ax.spines['top'].set_position(('data', 16000000))
    ax.spines['left'].set_bounds(0, 16000000)
    ax.spines['right'].set_bounds(0, 16000000)
    xnames = [i[0] for i in res_d]
    values = [i[1] for i in res_d]
    plt.axhline(y=608*608*3, color='r', linestyle='--')
    plt.text(-4, 608*608*3, 'input', color='r')
    x = np.arange(len(xnames))
    plt.bar(x, values, hatch='\\')
    for i in range(len(xnames)):
        plt.text(i, values[i]+3000000, '\%'+str(int(values[i]/(608*608*3)*100)), rotation=90)
    plt.xticks(x, xnames, rotation=90)
    plt.ticklabel_format(useOffset=False, axis='y', style='plain')
    plt.savefig('C:\\d\\diplomarbeit\\DA\\Figures\\params_num.pdf',
                format='pdf', dpi=1000)
    plt.show()

def plot_layers_info():
    plt.rcParams.update({'font.size': 15})
    # tex.setup(width=1, l=0.1, r=0.9, b=0.1, t=0.9)
    names = ["conv_1", "bn_1", "relu_1", "max_1", 
             "conv_2", "bn_2", "relu_2", "max_2",
             "conv_3", "bn_3", "relu_3", "conv_4",
             "bn_4", "relu_4", "conv_5", "bn_5",
             "relu_5", "max_3", "Mkl2Tf"][0:-1]
    mems = np.array([47.32, 4.64, 0, 14.79, 23.81, 47.33, 0, 7.39, 14.42, 
        23.67, 0, 5.98, 11.84, 0, 12.42, 23.67, 0, 3.7, 307.71])[0:-1]
    time = np.array([8.812, 39.694, 7.067, 24.514, 27.631, 20.458, 3.510, 14.988,
        27.894, 9.561, 1.639, 8.909, 4.581, 0.669, 31.349, 9.164, 1.645, 6.496, 66.338])[0:-1]
    x = np.arange(0, 54, 3)
    fig = plt.figure()
    ax1 = plt.subplot(211)
    ax1.bar(x, mems, width=2, color='b', align='center')
    # for a, b in zip(x, mems):
    #     ax1.text(a-1.3, b+1, str(int(np.round(b, 1))))
    ax1.set_xticks([])
    ax1.grid()
    # ax.set_xticklabels(names, rotation=90)
    ax1.set_ylabel("Memory (MB)")
    ax2 = plt.subplot(212)
    ax2.bar(x, time, width=2, color='r', align='center')
    # for a, b in zip(x, time):
    #     ax2.text(a-1.3, b+1, str(int(np.round(b, 1))))
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=90)
    ax2.set_ylabel("Time (ms)")
    plt.gca().yaxis.grid(True)
    fig.set_size_inches(5, 8)
    plt.subplots_adjust(hspace=0.1)
    plt.savefig('C:\\d\\diplomarbeit\\DA\\Figures\\plot_layers_info.pdf',
                format='pdf', dpi=1000)
    plt.show()

    '''
    ax.bar(x, time, color='r', align='center')
    
    for a, b in zip(x, time):
        if b != 0:
            ax.text(a-1, b, str(np.round(b, 1)))
    
    # ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_yticks([], [])
    plt.savefig('C:\\d\\diplomarbeit\\DA\\Figures\\plot_layers_info.pdf',
                format='pdf', dpi=1000)
    plt.show()
    '''


### accuracy of webp and jpeg
# plot_accu_quality()
### compresison ratio of webp and jpeg
# plot_ratio_quality()
### jpeg or webp performance plot
# accu_ratio('webp')
### compressibility analysis
# plot_compressibility_approaches()
### batching
# intra_speed_ratio()
### compression ratio over AP
# plot_intra_ratio_ap()
### plot accuracy when pruning small values
# plot_prune_ap()
### plot compression performance over AP of all image based compression methods
# plot_image_based_all(base='raw')
### plot compression performance over AP of all data based compression methods
# plot_data_based_all('jpg')
### feature maps space analysis
# plot_fmaps_space_analysis()
### different sorting strategies for h264
# plot_sort_h264()
### png vs zlib
# plot_png_zlib()
### params number
# plot_params_summary()
plot_layers_info()

### ----------------------------
# test()
# test_batched_h264()
