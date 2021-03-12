# import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from xml.dom import minidom
from datetime import datetime
import glob
import os
import sys
import time
# from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import tensorflow as tf
from collections import Counter
from barcode import BarCode




class Classification_infer():

    def __init__(self, image, info, model_path, label_path, mode):
        self.image = image
        self.info = info
        self.model_path = model_path
        self.label_path = label_path
        self.mode = mode

    def load_model(self):
        os.environ["CUDA_VISIBLE_DEVICES"]="0"
        main_model = tf.keras.models.load_model(self.model_path)
        
        return main_model

    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #   try:
    #     for gpu in gpus:
    #         tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10000)])
    #   except RuntimeError as e:
    #     print(e)

    def init_model(self, test_img, main_model, empty_model):
        img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = np.expand_dims(img, axis=0)
        #empty_model.predict(img)
        main_model.predict(img)


    def crop_image(self, image, boxes, resize=None, save_path=None):
        #print(self.image)
        images = list(map(lambda b : image[b[1]:b[3], b[0]:b[2]], boxes)) 
        
        if str(type(resize)) == "<class 'tuple'>":
            try:
                images = list(map(lambda i: preprocess_input(cv2.resize(i, dsize=resize, interpolation=cv2.INTER_LINEAR)), images))
            except Exception as e:
                print(str(e))
        return images


    def get_label(self):
        df = pd.read_csv(self.label_path, sep = ' ', index_col=False, header=None)
        class_name = df[0].tolist()
        class_name = sorted(class_name)
        return class_name

    def predict(self, main_model, CLASS_NAMES, images, ori_images):
        final_result = []
        for image, ori_image in zip(images, ori_images):
            img = np.expand_dims(image, axis=0)
            #img = preprocess_input(img)
            
            main_pred = main_model.predict(img)
            
            rst = CLASS_NAMES[np.argmax(main_pred)]
            if rst == 'bottom':
                rst = BarCode(ori_image).decode()

            #os.system('clear')
            final_result.append(rst)
        return final_result

    def inference(self, model):

        print('Model Load.................')
        main_model = model
        # get label name
        CLASS_NAMES = self.get_label()

        #if self.mode == 'ht':
        
            #image = cv2.imread(self.info['image_name'])
        image = self.image
        image = cv2.resize(image, (960, 960))
    
        if self.mode == 'ht':
            images = self.crop_image(image, self.info['section_1']['front_corr'], (224, 224))
            ori_images = self.crop_image(image, self.info['section_1']['front_corr'], (0, 0))

            final_result = self.predict(main_model = main_model, 
                                        CLASS_NAMES = CLASS_NAMES, 
                                        images = images, 
                                        ori_images = ori_images)
            print(final_result)
            return final_result


        elif self.mode == 'normal':
            final_result = []
            images_sec1 = self.crop_image(image, self.info['section_1']['front_corr'], (224, 224))
            ori_images_sec1 = self.crop_image(image, self.info['section_1']['front_corr'], (0, 0))

            images_sec2 = self.crop_image(image, self.info['section_2']['front_corr'], (224, 224))
            ori_images_sec2 = self.crop_image(image, self.info['section_2']['front_corr'], (0, 0))

            sec1_result = self.predict(main_model = main_model, 
                                        CLASS_NAMES = CLASS_NAMES, 
                                        images = images_sec1, 
                                        ori_images = ori_images_sec1)

            sec2_result = self.predict(main_model = main_model, 
                                        CLASS_NAMES = CLASS_NAMES, 
                                        images = images_sec2, 
                                        ori_images = ori_images_sec2)
            final_result.extend([sec1_result, sec2_result])
            print('front_cls_result : ', final_result)
            return final_result


#MODE = 'ht'  # single or dual

# info = {'direction': 'center',
#  'floor': '1',
#  'image_name': './test_images_ht/ht_1_center.jpg',
#  'section_1': {'front_corr': [(143, 850, 901, 933),
#                               (139, 754, 909, 835),
#                               (184, 668, 862, 743)],
#                'front_count': 3,
#                'total_corr': [(143, 850, 901, 933),
#                               (139, 754, 909, 835),
#                               (184, 668, 862, 743),
#                               (232, 463, 782, 519),
#                               (252, 410, 764, 459),
#                               (225, 524, 810, 582),
#                               (273, 317, 739, 354),
#                               (293, 237, 713, 269),
#                               (257, 362, 756, 402),
#                               (284, 275, 726, 309),
#                               (202, 591, 840, 657)],
#                'total_count': 11},
#  'total_count': 13}

# info = {'direction': 'right',
#  'floor': '8',
#  'image_name': './test_images/time_8_right.jpg',
#  'section_1': {'front_corr': [],
#                'front_count': 0,
#                'total_corr': [],
#                'total_count': 0},
#  'section_2': {'front_corr': [(471, 689, 738, 843), (476, 558, 703, 678)],
#                'front_count': 2,
#                'total_corr': [(471, 689, 738, 843), (476, 558, 703, 678)],
#                'total_count': 2},
#  'total_count': 4}



# MODE = 'normal'  # single or dual

# # get file path
# image = cv2.imread(info['image_name'])
# # label
# main_label_file = "./models/cls/ciga_v1_label.txt"

# # model
# main_model_path = "./models/cls/ciga_v1.h5"

# C = Classification_infer(image = image,
#                         info = info,
#                         model_path = main_model_path,
#                         label_path = main_label_file,
#                         mode = MODE)
# C.inference()

# main_label_file = project_path + "./models/cls/ciga_v1_label.txt"

# # model
# main_model_path = project_path + "./models/cls/ciga_v1.h5"