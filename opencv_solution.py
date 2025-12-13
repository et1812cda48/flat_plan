import numpy as np
import json
import cv2
import os

class SchemeProcessing:
    def __init__(self, file_path_json='coordinates_of_walls.json'):
        with open(file_path_json, 'w') as file:
            json.dump([], file)
        self.__file_path = file_path_json

    def __call__(self, scheme_obj):
        if isinstance(scheme_obj, list):
            for scheme_path in scheme_obj:
                scheme = self.__scheme_processing(scheme_path)
                coords = self.__get_coords(scheme)
                self.__save_coords(scheme_path, coords)
        else:
            scheme = self.__scheme_processing(scheme_obj)
            coords = self.__get_coords(scheme)
            self.__save_coords(scheme_obj, coords)

    @staticmethod
    def __scheme_processing(scheme_path):
        scheme = cv2.imread(scheme_path)
        '''scheme = cv2.cvtColor(scheme, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        walls = cv2.morphologyEx(cv2.adaptiveThreshold(clahe.apply(cv2.GaussianBlur(scheme, (3, 3), 0), 0), 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 5), cv2.MORPH_CLOSE, np.ones((3, 3)), iterations=1)'''
        _, th = cv2.threshold(scheme, 200, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        walls = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
        return walls

    @staticmethod
    def __get_coords(scheme, delta=10):
        edges = cv2.Canny(scheme, 20, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        coords = []
        for contour in contours:
            points = contour[:, 0, :]
            n_points = len(points)
            if n_points < 2:
                continue
            for i in range(n_points - 1):
                x0, y0 = points[i]
                x1, y1 = points[i + 1]
                dist = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
                if dist >= delta:
                    coords.append([[int(x0), int(y0)], [int(x1), int(y1)]])
            if n_points > 2:
                x0, y0 = points[-1]
                x1, y1 = points[0]
                dist = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
                if dist >= delta:
                    coords.append([[int(x0), int(y0)], [int(x1), int(y1)]])
        return coords

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

if __name__ == '__main__':
    file_list = os.listdir('schemes')
    Analyzator = SchemeProcessing()
    for file_name in file_list:
        print(f'Scheme named \"{file_name}\" is processing...')
        Analyzator(f'schemes/{file_name}')
    print('All schemes were processed.')