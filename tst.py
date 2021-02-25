from ctypes import *
import random
import os
import cv2
import time
import darknet
import argparse
from threading import Thread, enumerate
from queue import Queue
import datetime
from glob import glob

'''
2021.02.21
By JwMudfish

Yolo Inference 후 좌표 변환 해주는 코드
주의 : 기본 width, height - 960 (모델 Input 크기로 들어감)

'''


class ImageInfer():
    def __init__(self, weight_file, config_file, data_file, thresh_hold, image_path):
        self.weight_file = weight_file
        self.config_file = config_file
        self.data_file = data_file
        self.thresh_hold = thresh_hold
        self.image_path = image_path
        #self.image_width = image_width
        #self.image_height = image_height

    def bbox2points(self, bbox):
        x, y, w, h = bbox
        xmin = int(round(x - (w / 2)))
        xmax = int(round(x + (w / 2)))
        ymin = int(round(y - (h / 2)))
        ymax = int(round(y + (h / 2)))
        return xmin, ymin, xmax, ymax

    def get_corr(self):
        network, class_names, class_colors = darknet.load_network(self.config_file, self.data_file, self.weight_file, batch_size=1)
        width = darknet.network_width(network)
        height = darknet.network_height(network)
        darknet_image = darknet.make_image(width, height, 3)

        frame = cv2.imread(self.image_path)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height), interpolation=cv2.INTER_LINEAR)
        darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())

        detections = darknet.detect_image(network, class_names, darknet_image, thresh=self.thresh_hold, hier_thresh=.5, nms=.45)

        voc = []
        for label, prob, corr in detections:
            i = self.bbox2points(corr)
            voc.append(i)

        return voc

    def get_multi_corr(self, image_folder):
        network, class_names, class_colors = darknet.load_network(self.config_file, self.data_file, self.weight_file, batch_size=1)
        width = darknet.network_width(network)
        height = darknet.network_height(network)
        darknet_image = darknet.make_image(width, height, 3)

        infer_images = glob(os.path.join(image_folder, '*.jpg'))
        #print(infer_images)
        voc_rst = []
        for img in infer_images:
            frame = cv2.imread(img)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (width, height), interpolation=cv2.INTER_LINEAR)
            darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())

            detections = darknet.detect_image(network, class_names, darknet_image, thresh=self.thresh_hold, hier_thresh=.5, nms=.45)

            voc = []
            for label, prob, corr in detections:
                i = self.bbox2points(corr)
                voc.append(i)
            
            voc_rst.append({'image_name' : img, 'corr' : voc})
        return voc_rst


# inf = ImageInfer(weight_file = "/home/perth/Desktop/personal_project/yolov4/darknet/files/yolov4-custom_cigar_box_last.weights",
#                 config_file = "/home/perth/Desktop/personal_project/yolov4/darknet/files/yolov4-custom_cigar_box.cfg",
#                 data_file = "/home/perth/Desktop/personal_project/yolov4/darknet/files/cigar_box_obj.data",
#                 thresh_hold = .7,
#                 image_path = './image/main6.jpg')

# rst = inf.get_multi_corr('./test_images')

# print(rst)

'''
##############################################################
weight_file = "/home/perth/Desktop/personal_project/yolov4/darknet/files/yolov4-custom_cigar_box_last.weights"
config_file = "/home/perth/Desktop/personal_project/yolov4/darknet/files/yolov4-custom_cigar_box.cfg"
data_file = "/home/perth/Desktop/personal_project/yolov4/darknet/files/cigar_box_obj.data"
thresh_hold = .7

image_path = './image/main6.jpg'

network, class_names, class_colors = darknet.load_network(config_file, data_file, weight_file, batch_size=1)

###############################################################################################

frame_width = int(960)
frame_height = int(960)

##############################################################################################
width = darknet.network_width(network)
height = darknet.network_height(network)
darknet_image = darknet.make_image(width, height, 3)

frame = cv2.imread(image_path)
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame_resized = cv2.resize(frame_rgb, (width, height), interpolation=cv2.INTER_LINEAR)
darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())

detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh_hold, hier_thresh=.5, nms=.45)

voc = []
for label, prob, corr in detections:
    i = bbox2points(corr)
    voc.append(i)
'''