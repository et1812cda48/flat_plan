from transformers import DetrForObjectDetection
from ultralytics import YOLO
import easyocr
import numpy as np
import json
import cv2
import os

class SchemeSegmentation:
    def __init__(self, model_name='yolo11n-seg.pt', languages='en', file_name_json='coordinates_of_walls.json'):
        self.__file_path = file_name_json
        with open(file_name_json, 'w') as file:
            json.dump([], file)
        self.__yolo = YOLO(f'models/{model_name}')
        self.__transformer = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        if not isinstance(languages, list): languages = [languages]
        self.__ocr = easyocr.Reader(languages)

    def __call__(self, scheme_path):
        scheme = self.__scheme_processing(scheme_path)

    @staticmethod
    def __scheme_processing(scheme_path):
        scheme = cv2.imread(scheme_path)
        scheme = cv2.cvtColor(scheme, cv2.COLOR_BGR2GRAY)
        return scheme

    def __save_coords(self, scheme_name, coords):
        data = {"meta": {"source": scheme_name}, "walls": []}
        for i in range(len(coords)):
            wall = {"id": f"w{i+1}", "points": coords[i]}
            data["walls"].append(wall)
        with open(self.__file_path, 'r+') as file:
            info = json.load(file)
            info.append(data)
            file.seek(0)
            json.dump(info, file)