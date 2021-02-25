#from box_utils import BoxUtils
import os
import cv2
import easyocr
from ctypes import *
import random
from tst import ImageInfer
import pprint

'''
<카운트로직>
 - 일반담배 2개
    -> 왼쪽
    -> 오른쪽
 - 히트 3개
   - 왼쪽
   - 중앙
   - 오른쪽

1. 박스 면적 계산 후, 이상치 제거 : 이상치는 어떻게 판단할 것인지?
2. ocr 넣어봄 -> 뒤집힌 케이스는 잡아내기 어려움


이미지 파일 저장 형식 time_floor_lcr.jpg

'''


def center_point(corr, xy = 'x'):
    if xy == 'x':
        min_corr, max_corr = corr[0], corr[2]
        center_point = (max_corr + min_corr) / 2
    elif xy == 'y':
        min_corr, max_corr = corr[1], corr[3]
        center_point = (max_corr + min_corr) / 2
    return int(center_point)

def quickSort(x):
    if len(x) <= 1:
        return x
    pivot = x[len(x)//2]
    #print('pivot : ',pivot)
    left,right,equal =[],[],[]
    for a in x:
        #print(a)
        if a < pivot:
            left.append(a)
        elif a > pivot:
            right.append(a)
        else:
            equal.append(a)
    return quickSort(left) + equal + quickSort(right)


def get_front_corr(section, num, total=False):
    rst = sorted(section, key=lambda x : center_point(x, xy='y'), reverse=True)
    if total == True:
        rst = rst
    else:
        rst = rst[:num]

    return rst

def crop_image(image, boxes, save_path=None, labels=None, resize=None):
        seed_image = image
        images = list(map(lambda b : image[b[1]:b[3], b[0]:b[2]], boxes))
        images = list(map(lambda i : cv2.resize(i, resize), images))

        # num = 0            
        # for img, label in zip(images, labels):
        #     num = num + 1
            #cv2.imwrite('{}/{}/{}_{}_{}.jpg'.format(save_path,label,today,label, num), img)
            #cv2.imwrite('{}/{}/{}_{}.jpg'.format(save_path,label,today,label), img)
        return images


# 개수 카운트 해야하는 영역만 골라서 담기
def in_line(x,y,lr):
    # x = x
    # y = y
    if lr == 'left':
        x1, x2 = x, WIDTH
        y1, y2 = y, y+1

        pt1 = R_PNT, 0
        pt2 = WIDTH, HEIGHT

        imgRect = (x,y,x2-x1, y2-y1)
        retval, rpt1, rpt2 = cv2.clipLine(imgRect, pt1, pt2)

    elif lr == 'right':
        x1, x2 = x, WIDTH
        y1, y2 = y, y+1

        pt1 = L_PNT, 0
        pt2 = 0, HEIGHT

        imgRect = (x,y,x2-x1, y2-y1)
        retval, rpt1, rpt2 = cv2.clipLine(imgRect, pt1, pt2)

    return retval


def split_normal_section(lr):

    s1 = int(WIDTH // 2)

    section_1 = []
    section_2 = []

    for i in corr:
        x_corr = center_point(i, xy = 'x')
        y_corr = center_point(i, xy = 'y')
        
        if lr == 'left':
            if x_corr >= 0 and x_corr < s1:
                section_1.append(i)
            else:
                tf = in_line(x = x_corr, y = y_corr, lr = CAM)
                if tf == True:
                    section_2.append(i)

        elif lr == 'right':
            #if x_corr <= WIDTH and x_corr > s1:
            if x_corr > s1:
                section_2.append(i)
            else:
                tf = in_line(x = x_corr, y = y_corr, lr = CAM)  # right
                if tf == False:
                    section_1.append(i)

        elif lr == 'center':  # 히츠 중앙카메라일 경우 로직 추가해야 함
            pass
    return section_1, section_2
   

def draw_box(frame, box):
    for i in box:
        cv2.rectangle(frame, (i[0],i[1]), (i[2], i[3]), (0,0,255), 2)
        #cv2.putText(frame, calc_area(i), (i[0], i[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    return frame

def calc_area(corr):
    w = abs(corr[2] - corr[0])
    h = abs(corr[3] - corr[1])
    wh = int(w * h)
    return str(wh)

def use_ocr(image):
    reader = easyocr.Reader(['en'],gpu=True) # need to run only once to load model into memory
    result = reader.readtext(image)
    return result

##########################################################################################################

# 이미지 파일 형식은 floor_lcr
IMAGE_PATH = './image/ht_center.jpg'
CAM = 'right'

THRESH_HOLD = .7

WIDTH = 960
HEIGHT = 960
R_PNT = 620
L_PNT = 340


##################################################################################################

inf = ImageInfer(weight_file = "/home/perth/Desktop/personal_project/yolov4/darknet/files/yolov4-custom_cigar_box_last.weights",
                config_file = "/home/perth/Desktop/personal_project/yolov4/darknet/files/yolov4-custom_cigar_box.cfg",
                data_file = "/home/perth/Desktop/personal_project/yolov4/darknet/files/cigar_box_obj.data",
                thresh_hold = THRESH_HOLD,
                image_path = IMAGE_PATH)

corr = inf.get_corr()
print(corr)
#corrs = inf.get_multi_corr(image_folder = './test_images')
#print('총 박스 수 : ',len(corr))

image = cv2.imread(IMAGE_PATH)
image = cv2.resize(image, (WIDTH, HEIGHT))

section_1 = split_normal_section(lr = CAM)[0]
section_2 = split_normal_section(lr = CAM)[1]

section_1_count = len(split_normal_section(lr = CAM)[0])
section_2_count = len(split_normal_section(lr = CAM)[1])

section_1_front_corr = get_front_corr(section_1, num=3)
section_2_front_corr = get_front_corr(section_2, num=3)

# print(section_1)

# print('section_1 담배 수 : ', len(split_normal_section(lr = CAM)[0]))
# print(get_front_corr(section_1, num=3))

# print('section_2 담배 수 : ', len(split_normal_section(lr = CAM)[1]))
# print(get_front_corr(section_2, num=3))

images_corr_1 = get_front_corr(split_normal_section(lr = CAM)[0], num = 3)
images_corr_2 = get_front_corr(split_normal_section(lr = CAM)[1], num = 3)


final_result = {'image_name' : IMAGE_PATH,
                'total_count' : len(corr),
                'section_1' : { 'total_count' : section_1_count,
                                'total_corr' : section_1,
                                'front_count' : len(section_1_front_corr),
                                'front_corr' : section_1_front_corr },
                
                'section_2' : { 'total_count' : section_2_count,
                                'total_corr' : section_2,
                                'front_count' : len(section_2_front_corr),
                                'front_corr' : section_2_front_corr }
                                }

pprint.pprint(final_result)
#cv2.rectangle(image, (s1,0), (s1+1, HEIGHT), (0,0,255), 2)
#cv2.rectangle(image, (s2,0), (s2+1, HEIGHT), (0,0,255), 2)


####################  Painting Section  ###################################
if CAM == 'left':
    line = cv2.line(image, (R_PNT,0), (WIDTH, HEIGHT), (0,255,0), 4)

elif CAM == 'right':
    line = cv2.line(image, (L_PNT,0), (0, HEIGHT), (0,255,0), 4)

# print(images_corr_1)
#print(calc_area(corr[1]))


# images = crop_image(image = image, boxes = images_corr_1, resize=(224,224))
# #images_1 = crop_image(image = image, boxes = images_corr_1, resize=(224,224))
# sec_1_img = cv2.vconcat([images[0], images[1], images[2]])

# images = crop_image(image = image, boxes = images_corr_2, resize=(224,224))
# sec_2_img = cv2.vconcat([images[0], images[1], images[2]])


# concated_img = cv2.hconcat([sec_1_img, sec_2_img])

# 박스 그리기
image = draw_box(image, corr)


cv2.namedWindow('b_img', cv2.WINDOW_NORMAL)
cv2.resizeWindow('b_img', 960,960)
cv2.imshow('b_img', image)

# cv2.namedWindow('cc', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('cc', 800,600)
# cv2.imshow('cc', concated_img)

cv2.waitKey(0)
cv2.destroyAllWindows()