#from box_utils import BoxUtils
import os
import cv2
import easyocr
from ctypes import *
import random
from tst import ImageInfer
import pprint
from classification import Classification_infer
import tensorflow as tf

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
        #print('x : ', corr)
        min_corr, max_corr = corr[0], corr[2]
        center_point = (max_corr + min_corr) / 2
    elif xy == 'y':
        #print('y : ', corr)
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


def split_normal_section(lr, floor_mode):


    if floor_mode == 'normal':  # 일반 담배층일 경우
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

    else:
        s1 = int(WIDTH // 3)
        s2 = int(s1 * 2)

        section_1 = []
    
        for i in corr:
            x_corr = center_point(i, xy = 'x')
            y_corr = center_point(i, xy = 'y')

            if x_corr > s1 and x_corr < s2:
                section_1.append(i)

        return section_1



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

def load_model(model_path):
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    main_model = tf.keras.models.load_model(model_path)
    
    return main_model

##########################################################################################################

# 이미지 파일 형식은 floor_lcr
IMAGE_PATH = './image/main14.jpg'
#CAM = 'left'
FLOOR_MODE = 'ht'  # ht  #normal
THRESH_HOLD = .5

WIDTH = 960
HEIGHT = 960
R_PNT = 620
L_PNT = 340

WEIGHT_FILE = "/home/perth/Desktop/personal_project/1.ciga_detection/models/detection/yolov4-cigar_box_2021_last.weights"
CONFIG_FILE = '/home/perth/Desktop/personal_project/1.ciga_detection/models/detection/yolov4-cigar_box_2021.cfg'
DATA_FILE = '/home/perth/Desktop/personal_project/1.ciga_detection/models/detection/cigar_box_2021_obj.data'

main_label_file = "./models/cls/ciga_v1_label.txt"

# model
main_model_path = "./models/cls/ciga_v1.h5"

##################################################################################################

# inf = ImageInfer(weight_file = "/home/perth/Desktop/personal_project/yolov4/darknet/files/yolov4-custom_cigar_box_last.weights",
#                 config_file = "/home/perth/Desktop/personal_project/yolov4/darknet/files/yolov4-custom_cigar_box.cfg",
#                 data_file = "/home/perth/Desktop/personal_project/yolov4/darknet/files/cigar_box_obj.data",
#                 thresh_hold = THRESH_HOLD,
#                 image_path = IMAGE_PATH)

inf = ImageInfer(weight_file = WEIGHT_FILE,
                config_file = CONFIG_FILE,
                data_file = DATA_FILE,
                thresh_hold = THRESH_HOLD,
                image_path = IMAGE_PATH)




corrs = inf.get_multi_corr(image_folder = './test_images_ht')
cls_model = load_model(model_path = main_model_path)

#print('총 박스 수 : ',len(corr))
print(corrs)
for img in corrs:
    corr = img['corr']
    if img['image_name'].split('_')[-1] == 'left.jpg':
        CAM = 'left'

    else:
        CAM = 'right'

    if FLOOR_MODE == 'normal':    
        image = cv2.imread(img['image_name'])
        image = cv2.resize(image, (WIDTH, HEIGHT))

        section_1 = split_normal_section(lr = CAM, floor_mode = FLOOR_MODE)[0]
        section_2 = split_normal_section(lr = CAM, floor_mode = FLOOR_MODE)[1]

        section_1_count = len(split_normal_section(lr = CAM, floor_mode = FLOOR_MODE)[0])
        section_2_count = len(split_normal_section(lr = CAM, floor_mode = FLOOR_MODE)[1])

        section_1_front_corr = get_front_corr(section_1, num=3)
        section_2_front_corr = get_front_corr(section_2, num=3)

        images_corr_1 = get_front_corr(split_normal_section(lr = CAM, floor_mode = FLOOR_MODE)[0], num = 3)
        images_corr_2 = get_front_corr(split_normal_section(lr = CAM, floor_mode = FLOOR_MODE)[1], num = 3)

        # Redis 붙이면 달라질 수 있음......
        final_result = {'image_name' : img['image_name'],
                        'floor' : img['image_name'].split('/')[2].split('_')[1],
                        'direction' : img['image_name'].split('/')[2].split('_')[2].split('.')[0],
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

    else:
        image = cv2.imread(img['image_name'])
        image = cv2.resize(image, (WIDTH, HEIGHT))

        section_1 = split_normal_section(lr = CAM, floor_mode = FLOOR_MODE)

        section_1_count = len(split_normal_section(lr = CAM, floor_mode = FLOOR_MODE))

        section_1_front_corr = get_front_corr(section_1, num=3)

        images_corr_1 = get_front_corr(split_normal_section(lr = CAM, floor_mode = FLOOR_MODE), num = 3)

        # Redis 붙이면 달라질 수 있음......
        final_result = {'image_name' : img['image_name'],
                        'floor' : img['image_name'].split('/')[2].split('_')[1],
                        'direction' : img['image_name'].split('/')[2].split('_')[2].split('.')[0],
                        'total_count' : len(corr),
                        'section_1' : { 'total_count' : section_1_count,
                                        'total_corr' : section_1,
                                        'front_count' : len(section_1_front_corr),
                                        'front_corr' : section_1_front_corr }}


    C = Classification_infer(image = image,
                            info = final_result,
                            model_path = main_model_path,
                            label_path = main_label_file,
                            mode = FLOOR_MODE)
    os.system('clear')
    C.inference(model = cls_model)
    pprint.pprint(final_result)
    print('-' * 80)

    ####################  Painting Section  ###################################
    if CAM == 'left':
        line = cv2.line(image, (R_PNT,0), (WIDTH, HEIGHT), (0,255,0), 4)

    elif CAM == 'right':
        line = cv2.line(image, (L_PNT,0), (0, HEIGHT), (0,255,0), 4)

    # 박스 그리기
    image = draw_box(image, corr)


    cv2.namedWindow('b_img', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('b_img', 960,960)
    cv2.imshow('b_img', image)

    # cv2.namedWindow('cc', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('cc', 800,600)
    ## cv2.imshow('cc', concated_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

