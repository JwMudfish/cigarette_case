#-*- coding: utf-8 -*-
'''
coding by JW_Mudfish
Version : 1.0 
Last Updated 2020.12.15
'''

import pandas as pd
import cv2
import numpy as np
import imutils
from datetime import datetime
import time
import os
import random

# sudo apt install v4l-utils
###############################
BRIGHTNESS = 10
C_TYPE = 'normal'  # 'heat'
save_dir = './cls_seed_images'
# LEFT_CAMERA_NUM = 0
# RIGHT_CAMERA_NUM = 2
LR_MODE = 'b'

frame_width = int(1920)
frame_height = int(1080)

################################

# 음료 classification용
def aug_image(image):
    from albumentations import (
        HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90, RandomGamma, VerticalFlip,
        Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue, 
        IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
        IAASharpen, IAAEmboss, Flip, OneOf, Compose, Rotate, RandomContrast, RandomBrightness, RandomCrop, Resize, OpticalDistortion
    )

    transforms = Compose([
            #Rotate(limit=30, p=0.5),
            #Rotate(limit=180, p=0.5),
            #RandomRotate90(p=1.0)
            #Transpose(p=1.0)
            # Resize(248,248, p=1),     # resize 후 크롭
            # RandomCrop(224,224, p=1),  # 위에꺼랑 세트
            
            OneOf([
            RandomContrast(p=1, limit=(-0.5,2)),   # -0.5 ~ 2 까지가 현장과 가장 비슷함  -- RandomBrightnessContrast
            RandomBrightness(p=1, limit=(-0.2,0.4)),
            RandomGamma(p=1, gamma_limit=(80,200)),
            ], p=1),
                
            # OneOf([
            #     Rotate(limit=30, p=0.3),
            #     RandomRotate90(p=0.3),
            #     VerticalFlip(p=0.3)
            # ], p=0.3),
        
            MotionBlur(p=0.2),   # 움직일때 흔들리는 것 같은 이미지
            #ShiftScaleRotate(shift_limit=0.001, scale_limit=0.1, rotate_limit=30, p=0.3, border_mode=1),
            #Resize(224,224, p=1),
            ],
            p=1)
    return transforms(image=image)['image']

def make_folder(label_dir):
    if not os.path.exists(save_dir +'/' + label_dir):
        os.makedirs(save_dir +'/' + label_dir)

def file_count(save_path):
    dir_count = sorted(os.listdir(save_path))
    
    result = []
    for i in dir_count:
        folder_name = i
        files = len(os.listdir(f'{save_path}/{i}'))
        result.append([folder_name, files])
    return result

def getDevicesList():
    devices_list = []

    result = os.popen('v4l2-ctl --list-devices').read()
    result_lists = result.split("\n\n")
    for result_list in result_lists:
        if result_list != '':
            result_list_2 = result_list.split('\n\t')
            devices_list.append(result_list_2[1][-1])
    return devices_list

def image_capture(image, save_path, c_type, lr, resize = None):
    seed_image = image
    make_folder(c_type)
    cv2.imwrite(f'{save_path}/{c_type}/{today}_{c_type}_{lr}.jpg', image)

    print("Image Capture 함수실행!!! ", f'{save_path}/{c_type}/{today}_{c_type}_{lr}.jpg')

def auged_image_capture(image, save_path, c_type, lr, resize = None):
    seed_image = image
    image = aug_image(image)
    make_folder(c_type)
    cv2.imwrite(f'{save_path}/{c_type}/aug_{today}_{c_type}_{lr}.jpg', image)

def nothing(x):
    pass


#active_cam = list(map(lambda x : x[-1], getDevicesList()))
active_cam = getDevicesList()
if len(active_cam) >= 3:
    active_cam.remove('0')

print(f'현재 활성화 되어있는 카메라는 {active_cam} 입니다.')

frame0 = cv2.VideoCapture(int(active_cam[0]))
frame1 = cv2.VideoCapture(int(active_cam[1]))


MJPG_CODEC = 1196444237.0 # MJPG


cv2.namedWindow('Interminds Train Image Collection Program by JW', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Interminds Train Image Collection Program by JW', frame_width,frame_height)

frame0.set(cv2.CAP_PROP_BRIGHTNESS, BRIGHTNESS)
frame0.set(cv2.CAP_PROP_FOURCC, MJPG_CODEC)
frame0.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
frame0.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
frame0.set(cv2.CAP_PROP_AUTOFOCUS, 0)

frame1.set(cv2.CAP_PROP_BRIGHTNESS, BRIGHTNESS) 
frame1.set(cv2.CAP_PROP_FOURCC, MJPG_CODEC)
frame1.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
frame1.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
frame1.set(cv2.CAP_PROP_AUTOFOCUS, 0)

cv2.createTrackbar('Brightness','Interminds Train Image Collection Program by JW', 100, 200, nothing)


while True:

    frame0.set(cv2.CAP_PROP_BRIGHTNESS, BRIGHTNESS)
    frame1.set(cv2.CAP_PROP_BRIGHTNESS, BRIGHTNESS)

    if LR_MODE == 'b':
        ret0, img0 = frame0.read()
        ret1, img00 = frame1.read()

    else:
        ret0, img00 = frame0.read()
        ret1, img0 = frame1.read()


    # ret0, img0 = frame0.read()
    # ret1, img00 = frame1.read()
    
    blank_image_1 = np.zeros((150, frame_width, 3), np.uint8)
    blank_image_2 = np.zeros((150, frame_width, 3), np.uint8)

    img1 = cv2.resize(img0,(1920,1080))
    img2 = cv2.resize(img00,(1920,1080))

    concated_img1 = cv2.vconcat([blank_image_1, img1])
    concated_img2 = cv2.vconcat([blank_image_2, img2])
    
    rst = cv2.hconcat([concated_img1, concated_img2])
    blank_image_down = np.zeros((505, rst.shape[1], 3), np.uint8)

    folder_text_size = -60
    h = 100
    for i in file_count(save_dir):
        
        cv2.putText(blank_image_down, f'{i[0]} : {i[1]} /', (folder_text_size + 70, h), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 1, cv2.LINE_AA)    
        folder_len = cv2.getTextSize(text=str(f'{i[0]} : {i[1]} / '), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, thickness=2)[0][0]
        folder_text_size = folder_len + folder_text_size
        
        if folder_text_size >= rst.shape[1] - 800:
            h = h + 50
            folder_text_size = -60


    cv2.putText(blank_image_down, 'Brightness : {}'.format(BRIGHTNESS), (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0,255,0), 2, cv2.LINE_AA)

    rst = cv2.vconcat([rst, blank_image_down])

    BRIGHTNESS = cv2.getTrackbarPos('Brightness', 'Interminds Train Image Collection Program by JW') - 100

    cv2.imshow('Interminds Train Image Collection Program by JW', rst)
    today = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
   
    ch = cv2.waitKey(1)

######################################################################################################################
    # 종료
    if ch == ord('q'):
	    break

    elif ch == ord('1'):
        image_1 = img1
        image_capture(image = image_1, save_path = save_dir, c_type = C_TYPE, lr = 'left')
    
    elif ch == ord('2'):
        image_2 = img2
        image_capture(image = image_2, save_path = save_dir, c_type = C_TYPE, lr = 'right')
    
    elif ch == ord('c'):
        image_1 = img1
        image_2 = img2

        image_capture(image = image_1, save_path = save_dir, c_type = C_TYPE, lr = 'left')
        auged_image_capture(image = image_1, save_path = save_dir, c_type = C_TYPE, lr = 'left')

        image_capture(image = image_2, save_path = save_dir, c_type = C_TYPE, lr = 'right')
        auged_image_capture(image = image_2, save_path = save_dir, c_type = C_TYPE, lr = 'right')

    elif ch == ord('l'):
        if LR_MODE == 'b':
            LR_MODE = 'a'
        else:
            LR_MODE = 'b'

frame0.release()
frame1.release()
cv2.destroyAllWindows()