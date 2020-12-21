from box_utils import BoxUtils
import os
import cv2
import easyocr

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


corr = BoxUtils.get_boxes('./xml')['1']
print('총 박스 수 : ',len(corr))

WIDTH = 1920
HEIGHT = 1080

image = cv2.imread('./image/main2.jpg')

# 카메라 각도에 따라서 section 수 자동으로 배정하는 로직은 하드웨어 나오고 짜야함
# s1 = int(WIDTH / 3)
# s2 = int(s1 * 2)

# 개수 카운트 해야하는 영역만 골라서 담기
def in_line(x,y):
    x = x
    y = y
    x1, x2 = x, WIDTH
    y1, y2 = y, y+1

    pt1 = 1300, 0
    pt2 = WIDTH, HEIGHT

    imgRect = (x,y,x2-x1, y2-y1)
    retval, rpt1, rpt2 = cv2.clipLine(imgRect, pt1, pt2)
    return retval


def split_normal_section():

    s1 = int(WIDTH // 2)
    s2 = int((WIDTH + s1) / 2.1)

    section_1 = []
    section_2 = []

    for i in corr:
        rst = center_point(i, xy='x')
        if rst >= 0 and rst < s1:
            section_1.append(i)
        else:
            tf = in_line(x =rst, y = center_point(i, xy='y'))
            if tf == True:
                section_2.append(i)
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


print('section_1 담배 수 : ', len(split_normal_section()[0]))
#print(get_front_corr(section_1, num=3))

print('section_2 담배 수 : ', len(split_normal_section()[1]))
#print(get_front_corr(section_2, num=3))

images_corr_1 = get_front_corr(split_normal_section()[0], num = 3)
images_corr_2 = get_front_corr(split_normal_section()[1], num = 3)

#cv2.rectangle(image, (s1,0), (s1+1, HEIGHT), (0,0,255), 2)
#cv2.rectangle(image, (s2,0), (s2+1, HEIGHT), (0,0,255), 2)


line = cv2.line(image, (1300,0), (WIDTH, HEIGHT), (0,255,0), 4)
print(images_corr_1[0])
print(calc_area(corr[1]))


images = crop_image(image = image, boxes = images_corr_1, resize=(224,224))
images_1 = crop_image(image = image, boxes = images_corr_1, resize=(224,224))
sec_1_img = cv2.vconcat([images[0], images[1], images[2]])

images = crop_image(image = image, boxes = images_corr_2, resize=(224,224))
sec_2_img = cv2.vconcat([images[0], images[1], images[2]])


concated_img = cv2.hconcat([sec_1_img, sec_2_img])

# 박스 그리기
image = draw_box(image, corr)


#cv2.imshow('img', concated_img)
#cv2.imshow('img', images_1[2])

cv2.namedWindow('b_img', cv2.WINDOW_NORMAL)
cv2.resizeWindow('b_img', 800,600)
cv2.imshow('b_img', image)

cv2.namedWindow('cc', cv2.WINDOW_NORMAL)
cv2.resizeWindow('cc', 800,600)
cv2.imshow('cc', concated_img)

cv2.waitKey(0)
cv2.destroyAllWindows()