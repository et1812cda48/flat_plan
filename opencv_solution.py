import numpy as np
import json
import cv2
import os

class SchemeProcessing:
    '''в этом методе происходит инициализация объекта класса и создание файла json'''
    def __init__(self, file_path_json='coordinates_of_walls.json'):
        with open(file_path_json, 'w') as file:
            json.dump([], file)
        self.__file_path = file_path_json

    '''этот метод обрабатывает путь или список путей до изображений схем'''
    def __call__(self, scheme_obj):
        if isinstance(scheme_obj, list):
            for scheme_path in scheme_obj:
                self.__make_iteration(scheme_path)
        else:
            self.__make_iteration(scheme_obj)

    '''этот метод содержит последовательность вызова остальных методов'''
    def __make_iteration(self, scheme_path):
        scheme = self.__scheme_processing(scheme_path)
        coords = self.__get_coords(scheme)
        self.__save_coords(scheme_path, coords)

    '''в этом методе обрабатывается схема квартиры и на нее накладываются различные фильтры'''
    @staticmethod
    def __scheme_processing(scheme_path):
        scheme = cv2.imread(scheme_path)
        scheme = cv2.cvtColor(scheme, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(scheme, 200, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        walls = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
        return walls

    '''в этом методе извлекаются координаты стен из изображения схемы'''
    def __get_coords(self, scheme, delta=10):
        coords = []
        edges = cv2.Canny(scheme, 10, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
        '''for i in range(1, scheme.shape[0] - 1):
            j = 1
            while j < scheme.shape[1] - 1:
                if scheme[i][j].all() == 255 or np.array_equal(scheme[i-1][j], scheme[i+1][j]) and np.array_equal(scheme[i][j-1], scheme[i][j+1]):
                    j += 1
                    continue
                (x1, y0), (x0, y1) = self.__cell_processing(scheme, i, j)
                coords.append([[x1, y0], [x0, y1]])
                j = y1'''
        return coords

    '''этот метод определяет границы зон и возращает их координаты'''
    @staticmethod
    def __cell_processing(scheme, x0, y0, step=1):
        first_status, current_status = scheme[x0-1:x0+2, y0-1:y0+2], None
        x, y = x0, y0
        while x < scheme.shape[0] - 1:
            x += step
            current_status = scheme[x-1:x+2, y0-1:y0+2]
            if not np.array_equal(first_status, current_status): break
        while y < scheme.shape[1] - 1:
            y += step
            current_status = scheme[x0-1:x0+2, y-1:y+2]
            if not np.array_equal(first_status, current_status): break
        return (x, y0), (x0, y)

    '''в этом методе сохраняются координаты в файле json'''
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