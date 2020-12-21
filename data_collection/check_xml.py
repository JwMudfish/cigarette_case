import xml.etree.ElementTree as ET
import os
import glob

class CheckXml():
    def __init__(self, image_path, label_path):
        self.image_path = image_path
        self.label_path = label_path

    def get_label_list(self):
        