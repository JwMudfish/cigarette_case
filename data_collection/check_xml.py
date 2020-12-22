import xml.etree.ElementTree as ET
import os
import glob

class CheckXml():
    def __init__(self, image_path, xml_path, c_type, predefined_class_path):
        self.image_path = image_path
        self.xml_path = xml_path
        self.predefined_class_path = predefined_class_path
        self.c_type = c_type

    def get_label_list(self):
        f = open(self.predefined_class_path, 'r')
        labels = f.readlines()
        f.close()

        label_list = []
        for label in labels:
            label = str(label.replace('\n',''))
            label_list.append(label)
        return label_list

    def get_root(self, file_path):
        tree = ET.parse(file_path)
        root = tree.getroot()
        return root

    def label_name_check(self):
        total_xml_list = glob.glob(self.xml_path + '/' + self.c_type + '/*')
        print('"1.fault label name checking..."')
        for xml_file in total_xml_list:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for obj in root.iter('object'):
                name = obj.findtext('name')

                if not name in self.get_label_list():
                    print(f'잘못된 라벨명 : {name} ||||  라벨파일 : {xml_file}')

        print('Done')
        print('-' * 30)
        return total_xml_list

    def compare_img_xml(self):
        image_list = os.listdir(os.path.join(self.image_path,self.c_type))
        image_list = list(map(lambda x : x.split('.')[0], image_list))

        xml_list = os.listdir(os.path.join(self.xml_path,self.c_type))
        xml_list = list(map(lambda x : x.split('.')[0], xml_list))

        
        if len(image_list) == len(xml_list):
                print('The number of label files is the same !!')
                print('파일 이름 같은지 확인 시작!!')
                print(set(image_list) - set(xml_list))

                print('파일 이름 같은지 확인 완료! 위에 뭐가 안뜨면 좋은거임')
        else:
            print(' >> The number of label files is different. Please check again !! <<')
            
            if len(image_list) > len(xml_list):
                print('There is no files in xml (image는 있지만 xml은 없음) : ', set(image_list) - set(xml_list))
            else:
                print('There is no files in image (xml은 있지만 image는 없음): ', set(xml_list) - set(image_list))

        #print(image_list)
        #print(xml_list)
        print(set(image_list) - set(xml_list))
        #print(set(xml_list) - set(image_list))

a = CheckXml(image_path = './image', xml_path = './xml', predefined_class_path = './label.txt', c_type='')
#print(a.get_label_list())
#print(a.label_name_check())
print(a.compare_img_xml())