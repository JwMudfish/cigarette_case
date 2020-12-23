import xml.etree.ElementTree as ET
import os
import glob

class CheckXml():
    def __init__(self, image_path, xml_path, c_type, predefined_class_path):
        self.image_path = image_path
        self.xml_path = xml_path
        self.predefined_class_path = predefined_class_path
        self.c_type = c_type

        self.xml_list = os.listdir(os.path.join(self.xml_path,self.c_type))
        self.total_xml_list = glob.glob(self.xml_path + '/' + self.c_type + '/*')

        self.image_list = os.listdir(os.path.join(self.image_path,self.c_type))


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
        # total_xml_list = glob.glob(self.xml_path + '/' + self.c_type + '/*')
        print('1.wrong label checking...')
        for xml_file in self.total_xml_list:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for obj in root.iter('object'):
                name = obj.findtext('name')

                if not name in self.get_label_list():
                    print(f'   잘못된 라벨명 : {name} ||||  라벨파일 : {xml_file}')

        return print('Done!!!')

    def compare_img_xml(self):
        print('-' * 60)
        print('2.이미지 파일, 라벨 파일 비교 시작')
        # image_list = os.listdir(os.path.join(self.image_path,self.c_type))
        image_list = list(map(lambda x : x.split('.')[0], self.image_list))

        #xml_list = os.listdir(os.path.join(self.xml_path,self.c_type))
        #print('xml_list', self.xml_list)
        xml_list = list(map(lambda x : x.split('.')[0], self.xml_list))

        if len(image_list) == len(xml_list):
                print('  파일 이름 같은지 확인 시작..')
                rst = set(image_list) - set(xml_list)
                if len(rst) == 0: 
                    print('  이미지, 라벨파일 수 같음, 이름 모두 같음 !!!!!!!')
                else:
                    print('  이미지, xml 파일 개수는 같음. 다른 파일명 존재')

        else:
            print('   The number of label files is different. Please check again !! <<')    
            if len(image_list) > len(xml_list):
                print('   xml 파일이 없음 (image는 있지만 xml은 없음) : ', set(image_list) - set(xml_list))
            else:
                print('   image 파일이 없음 (xml은 있지만 image는 없음): ', set(xml_list) - set(image_list))
        return print('Done !!!!!!!!!!!!')
        #print(set(image_list) - set(xml_list))

    def change_xml(self, wrong_label=False, correct_label=False):
        print('-' * 60)
        print('3.잘못된 라벨 변경 시작')
        for xml_file in self.total_xml_list:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            for elem in root.getiterator():
                elem.text = elem.text.replace(wrong_label, correct_label)
            tree.write(xml_file)
        print(f'   wrong_label : {wrong_label}, ||| correct_label : {correct_label} , 변경완료!!')


# a = CheckXml(image_path = './image', xml_path = './xml', predefined_class_path = './label.txt', c_type='')

#print(a.get_label_list())
#print(a.label_name_check())
# a.change_xml(wrong_label='aaa', correct_label='object')