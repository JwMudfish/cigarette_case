from classification import Classification_infer
import cv2

info = {'direction': 'right',
 'floor': '8',
 'image_name': './test_images/time_8_right.jpg',
 'section_1': {'front_corr': [],
               'front_count': 0,
               'total_corr': [],
               'total_count': 0},
 'section_2': {'front_corr': [(471, 689, 738, 843), (476, 558, 703, 678)],
               'front_count': 2,
               'total_corr': [(471, 689, 738, 843), (476, 558, 703, 678)],
               'total_count': 2},
 'total_count': 4}



MODE = 'normal'  # single or dual

# get file path
image = cv2.imread(info['image_name'])
# label
main_label_file = "./models/cls/ciga_v1_label.txt"

# model
main_model_path = "./models/cls/ciga_v1.h5"

C = Classification_infer(image = image,
                        info = info,
                        model_path = main_model_path,
                        label_path = main_label_file,
                        mode = MODE)
C.inference()