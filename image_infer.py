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

def bbox2points(bbox):
    """
    From bounding box yolo format
    to corner points cv2 rectangle
    """
    x, y, w, h = bbox
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

weight_file = "/home/perth/Desktop/personal_project/yolov4/darknet/files/yolov4-custom_cigar_box_last.weights"
config_file = "/home/perth/Desktop/personal_project/yolov4/darknet/files/yolov4-custom_cigar_box.cfg"
data_file = "/home/perth/Desktop/personal_project/yolov4/darknet/files/cigar_box_obj.data"
thresh_hold = .7

image_path = './image/main.jpg'

network, class_names, class_colors = darknet.load_network(config_file, data_file, weight_file, batch_size=1)

###############################################################################################
MJPG_CODEC = 1196444237.0 # MJPG
cap_AUTOFOCUS = 0
cap_FOCUS = 0


frame_width = int(960)
frame_height = int(960)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
# # cv2.namedWindow('inference', cv2.WINDOW_FREERATIO)
# # cv2.resizeWindow('inference', frame_width, frame_height)

# cap.set(cv2.CAP_PROP_BRIGHTNESS, 0)
# cap.set(cv2.CAP_PROP_FOURCC, MJPG_CODEC)
# #cap.set(cv2.CAP_PROP_AUTOFOCUS, cap_AUTOFOCUS)
# cap.set(cv2.CAP_PROP_FOCUS, cap_FOCUS)
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
    #cv2.rectangle(image, (i[0],i[1]), (i[2], i[3]), (0,0,255), 2)

print(voc)
# darknet.print_detections(detections)
#print(detections)
#voc = darknet.bbox2points

#image = darknet.draw_boxes(detections, frame_resized, class_colors)
image = frame_resized
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.putText(image, "Objects : " + str(len(detections)), (width-220, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
# image = cv2.resize(image, (1280, 720))
# print('NUM OF OBJECT :',  len(detections))

cv2.imshow("inference", image)

cv2.waitKey(0)
cv2.destroyAllWindows()