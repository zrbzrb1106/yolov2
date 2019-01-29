import numpy as np
from detect import *
import cv2
import json

if __name__ == "__main__":
    f = open('./cocoapi/annotations/instances_val2017.json')
    ann = json.load(f)
    f = open('./cocoapi/results/results_cut.json')
    res = json.load(f)
    print(0)