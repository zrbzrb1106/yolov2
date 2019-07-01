from memory_profiler import memory_usage
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    print("enter pid: ")
    pid = int(input())
    mem_usage = memory_usage(pid, interval=0.1, timeout=20) 
    plt.plot(mem_usage)
    plt.show()