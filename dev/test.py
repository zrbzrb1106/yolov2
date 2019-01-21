import numpy as np
from detect import *
import cv2

if __name__ == "__main__":
    with open("dev/0_output.bin", mode="rb") as f:
        filecontent = f.read()
    print(len(filecontent))
    data = np.frombuffer(filecontent, np.uint8)
    f_map = np.reshape(data[0: 76*76], (76, 76))
    tmp = np.zeros(shape=(76, 76))
    f_tmp = cv2.normalize(
                f_map, tmp, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    # f_map = np.expand_dims(f_map, axis=0)
    # get_feature_map(f_map, is_display=1)
    cv2.imshow("tmp", f_tmp)
    cv2.waitKey(0)