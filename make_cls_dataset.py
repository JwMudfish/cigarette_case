import os
import cv2
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pprint import pprint
from glob import glob

def make_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def crop_image(image, boxes, save_path, labels, resize=None):
        seed_image = image
        images = list(map(lambda b : image[b[1]+1:b[3]-1, b[0]+2:b[2]-1], boxes))
        images = list(map(lambda i : cv2.resize(i, resize), images))

        num = 0            
        for img, label in zip(images, labels):
            num = num + 1
            cv2.imwrite(f'{save_path}/{label}/{label}_{num}.jpg', img)
            #cv2.imwrite('{}/{}/{}_{}.jpg'.format(save_path,label,today,label), img)
        return images

def get_boxes(xml_file):
      #try:
    xml_path = os.path.join(xml_file)

    root_1 = minidom.parse(xml_path)

    name = root_1.getElementsByTagName('name')
    bnd_1 = root_1.getElementsByTagName('bndbox')

    result = []
    for i in range(len(bnd_1)):
        
        label = name[i].childNodes[0].nodeValue
        xmin = int(bnd_1[i].childNodes[1].childNodes[0].nodeValue)
        ymin = int(bnd_1[i].childNodes[3].childNodes[0].nodeValue)
        xmax = int(bnd_1[i].childNodes[5].childNodes[0].nodeValue)
        ymax = int(bnd_1[i].childNodes[7].childNodes[0].nodeValue)

        result.append({'label' : label, 'bndbox' : (xmin,ymin,xmax,ymax)})
    return result


data_path = './data/detection/1.seed/image'
xml_path = './data/detection/1.seed/xml'
save_path = './data/detection/output'

image_list = glob(data_path + '/*')
resize = (224,224)

num = 0
for img in image_list:
    image = cv2.imread(img)

    xml_name = img.split('/')[-1].split('.')[0] + '.xml'
    for xml in get_boxes(xml_path + '/' + xml_name):
        #print(xml)
        # for i in xml:
        #     print(i)
        b = xml['bndbox']
        label = xml['label']
        cropped_img = image[b[1]+1:b[3]-1, b[0]+2:b[2]-1]
        cropped_img = cv2.resize(cropped_img, resize)
        make_folder(f'{save_path}/{label}')
        cv2.imwrite(f'{save_path}/{label}/{label}_{num}.jpg', cropped_img)
        num = num + 1
