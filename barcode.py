import pyzbar.pyzbar as pyzbar
import cv2
from PIL import Image
import matplotlib.pyplot as plt


class BarCode():

    def __init__(self, image):
        self.image = image

    def decode(self):

        img = self.image
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = cv2.GaussianBlur(img, (0,0), 1.0)
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        decoded = pyzbar.decode(img)
        
        # rst = []
        # for d in decoded: 
        #     x, y, w, h = d.rect

        #     barcode_data = d.data.decode("utf-8")
        #     barcode_type = d.type

        #     #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        #     #print(barcode_data)
        #     rst.append(barcode_data)
        
        rst = decoded[0].data.decode("utf-8")
        return rst