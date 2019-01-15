# yolov2
A simple implementation of yolov2 based on tensorflow
Only for detection, training is not supported
The repo is based on https://github.com/KOD-Chen/YOLOv2-Tensorflow, I just
use the weights from official yolo page and for private usage
## HOW TO USE
### Run in Python
dependency: python3 tensorflow opencv

1. git clone https://github.com/thtrieu/darkflow/tree/master/darkflow
to install darkflow(follow the official guide)

2. download weights from https://pjreddie.com/media/files/yolov2.weights
and download cfg file from https://github.com/pjreddie/darknet/blob/master/cfg/yolov2.cfg

3. run flow --model cfg/yolo.cfg --load bin/yolo.weights --savepb
by which --model and --load specify the cfg and weights files downloaded

4. under built\_graph dictory will be the model files and just put them under 
the model dictory of this project(mkdir model)

5. run detect.py and enjoy

### Run in docker
make sure docker is installed, then

1. docker build -t yolov2 -f Dockerfile.yolov2

2. 5 minutes waiting and everything is there

3. docker run -it ID /bin/bash
