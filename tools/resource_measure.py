# -*- coding: utf-8 -*-

import psutil
import time


def get_percent(process):
    try:
        return process.cpu_percent()
    except AttributeError:
        return process.get_cpu_percent()


def get_memory(process):
    try:
        return process.memory_info()
    except AttributeError:
        return process.get_memory_info()


def monitor(pid, logfile=None, duration=0):
    pr = psutil.Process(pid)
    start_time = time.time()
    if logfile:
        f = open(logfile, 'w')
        f.write("# {0:12s} {1:12s} {2:12s} {3:12s}\n".format(
            'Elapsed time'.center(12),
            'CPU (%)'.center(12),
            'Real (MB)'.center(12),
            'Virtual (MB)'.center(12))
        )
    log = {}
    log['times'] = []
    log['cpu'] = []
    log['mem_real'] = []
    log['mem_virtual'] = []

    try:
        while True:
            current_time = time.time()
            pr_status = pr.status()

            if pr_status in [psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD]:
                print("Process finished ({0:.2f} seconds)"
                      .format(current_time - start_time))
                break

            if duration is not None and current_time - start_time > duration:
                break

            current_cpu = pr.cpu_percent(interval=1)
            print("*****************", current_cpu)
            current_mem = pr.memory_info()
            print("*****************", current_mem)
            if int(current_cpu) == 0:
                continue

            current_mem_real = current_mem.rss / 1024. ** 2
            current_mem_virtual = current_mem.vms / 1024. ** 2

            if logfile:
                f.write("{0:12.3f} {1:12.3f} {2:12.3f} {3:12.3f}\n".format(
                    current_time - start_time,
                    current_cpu,
                    current_mem_real,
                    current_mem_virtual))
                f.flush()

            log['times'].append(current_time - start_time)
            log['cpu'].append(current_cpu)
            log['mem_real'].append(current_mem_real)
            log['mem_virtual'].append(current_mem_virtual)
    except KeyboardInterrupt:
        pass

    if logfile:
        f.close()

    return log


if __name__ == "__main__":

    print("enter pid: ")
    pid = int(input())
    logfile = "./logs/log_yolo"
    dura = 20
    log = monitor(pid, logfile=logfile, duration=dura)

    mean_cpu = sum(log['cpu']) / float(len(log['cpu']))
    mean_mem_real = sum(log['mem_real']) / float(len(log['mem_real']))

    print("Average usage of cpu: {}, mem: {} MB in {} seconds".format(
        mean_cpu, mean_mem_real, dura))

    # inf_time = measure.inf

    # pre = inf_time['pre']
    # inf = inf_time['inf']
    # post = inf_time['post']

    # print("Average inference time ---- pre: {}, inf: {}, post: {}".format(
    #     sum(pre) / float(len(pre)),
    #     sum(inf) / float(len(inf)),
    #     sum(post) / float(len(post))
    # ))
