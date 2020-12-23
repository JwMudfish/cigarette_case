from check_xml import CheckXml

IMAGE_PATH = './image'
XML_PATH = './xml'
C_TYPE = ''
PREDEFINED_CLASS_PATH = './label.txt'

checkxml = CheckXml(image_path = IMAGE_PATH, 
           xml_path = XML_PATH,
           predefined_class_path = PREDEFINED_CLASS_PATH,
           c_type=C_TYPE)

print('<< 설정된 라벨 확인 >>')
print(checkxml.get_label_list())
print('-'*60)

checkxml.label_name_check()
checkxml.compare_img_xml()

# 잘못된 라벨이 있다면 사용!!
checkxml.change_xml(wrong_label='ob1ject', correct_label='object')