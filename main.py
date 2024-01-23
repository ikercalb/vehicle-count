import cv2
import numpy as np
import torch
from tracker import *
from datetime import datetime
import csv

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

cap = cv2.VideoCapture("Road traffic video for object recognition.mp4")
ir_csv = 'ir.csv'
count = 0
tracker = Tracker()

cv2.namedWindow('Deteccion de vehiculos')
area1 = [(290, 325), (465, 325), (465, 335), (290, 335)]
area2 = [(545, 340), (755, 340), (755, 355), (545, 355)]
ir = set()
venir = set()
datos_ir = ()

# Frecuencia objetivo de 30 fps
fps_target = 600
wait_time = int(1000 / fps_target)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020, 600))
    results = model(frame)
    bbox_list = []
    for index, rows in results.pandas().xyxy[0].iterrows():
        x = int(rows[0])
        y = int(rows[1])
        x1 = int(rows[2])
        y1 = int(rows[3])
        b = str(rows['name'])
        bbox_list.append([x, y, x1, y1, b])

    idx_bbox = tracker.update(bbox_list)

    for bbox in idx_bbox:
        x2, y2, x3, y3, id, b = bbox
        cv2.rectangle(frame, (x2, y2), (x3, y3), (0, 0, 255), 2)
        cv2.putText(frame, str(id), (x2, y2), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
        cv2.circle(frame, (x3, y3), 3, (0, 255, 0), -1)

        results = cv2.pointPolygonTest(np.array(area1, np.int32), ((x3, y3)), False)
        results1 = cv2.pointPolygonTest(np.array(area2, np.int32), ((x3, y3)), False)

        fecha_actual = datetime.now()

        if results > 0:
            datos_ir = (id, b, fecha_actual.strftime("%Y-%m-%d"), fecha_actual.strftime("%H:%M:%S"))
            if datos_ir not in ir:
                with open('ir.csv', 'a', newline='') as archivo_csv:
                    escritor = csv.writer(archivo_csv)
                    escritor.writerow(datos_ir)
                ir.add(datos_ir)

        if results1 > 0:
            datos_venir = (id, b, fecha_actual.strftime("%Y-%m-%d"), fecha_actual.strftime("%H:%M:%S"))
            if datos_venir not in venir:
                with open('venir.csv', 'a', newline='') as archivo_csv:
                    escritor = csv.writer(archivo_csv)
                    escritor.writerow(datos_venir)
                venir.add(datos_venir)


    cv2.polylines(frame, [np.array(area1, np.int32)], True, (0, 255, 255), 3)
    cv2.polylines(frame, [np.array(area2, np.int32)], True, (0, 255, 255), 3)

    a1 = len(ir)
    a2 = len(venir)

    cv2.putText(frame, str(a1), (549, 465), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
    cv2.putText(frame, str(a2), (804, 411), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
    cv2.imshow("Deteccion de vehiculos", frame)

    if cv2.waitKey(wait_time) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
