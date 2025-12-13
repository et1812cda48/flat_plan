import json
import cv2
import os

scheme_folder = 'detected_schemes'

if not os.path.exists(scheme_folder):
    os.mkdir(scheme_folder)
with open('coordinates_of_walls.json', 'r') as file:
    table = json.load(file)

for data in table:
    scheme = cv2.imread(data['meta']['source'])
    for line in data['walls']:
        [x1, y1], [x2, y2] = line['points']
        cv2.line(scheme, (x1, y1), (x2, y2), (0, 0, 255), 1)
    cv2.imwrite('{}/{}'.format(scheme_folder, data['meta']['source'].split('/')[-1]), scheme)
    '''cv2.imshow('Detected scheme', scheme)
    cv2.waitKey(0)
cv2.destroyAllWindows()'''