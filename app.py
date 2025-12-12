import matplotlib.pyplot as plt
import numpy as np
import math
import json
import cv2
import os

def scheme_proccessing(scheme_path):
    scheme = cv2.imread(scheme_path)
    scheme = cv2.cvtColor(scheme, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    '''_, th = cv2.threshold(scheme, 200, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    walls = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)'''
    walls = cv2.morphologyEx(
    cv2.adaptiveThreshold(clahe.apply(cv2.GaussianBlur(scheme, (3, 3), 0), 0), 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 5), cv2.MORPH_CLOSE, np.ones((3, 3)), iterations=1)
    return walls

def get_coords(scheme, delta=10):
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

def save_coords(coords, file_path='coordinates_of_walls.json'):
    data = {"meta": {"source": file_path.split('/')[-1]}, "walls": []}
    for i in range(len(coords)):
        x1, y1, x2, y2 = list(map(int, coords[i]))
        wall = {"id": f"w{i+1}", "points": [[x1, y1], [x2, y2]]}
        data["walls"].append(wall)
    print(data)
    with open(file_path, 'w') as file:
        json.dump(data, file)

file_name = 'schemes/5.png'
scheme = scheme_proccessing(file_name)
coords = get_coords(scheme)
#save_coords(coords[:, 0])
scheme = cv2.imread(file_name)
for xyxy in coords:
    print(xyxy)
    try:
        x1, y1, x2, y2 = xyxy[0]
        cv2.line(scheme, (x1, y1), (x2, y2), (255, 0, 0), 1)
    except:
        cv2.polylines(scheme, [np.array(xyxy, dtype=np.int32)], True, (255, 0, 0), 1)
plt.title(file_name)
plt.imshow(scheme)
plt.axis('off')
plt.show()