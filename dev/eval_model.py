import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab


def get_mAPs():
    annType = 'bbox'
    prefix = 'instances'
    dataDir = '../'
    dataType = 'val2017'
    annFile = '%s/annotations/%s_%s.json' % (dataDir, prefix, dataType)
    resFile = '%s/results/%s_%s_fake%s100_results.json' % (
        dataDir, prefix, dataType, annType)
    cocoGt = COCO(annFile)
    cocoDt = COCO(resFile)
    cocoEval = COCOeval(cocoGt,cocoDt,annType)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


if __name__ == "__main__":
    get_mAPs()