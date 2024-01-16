import cv2
import numpy as np
import torch
from tracker import *
import datetime

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

cap = cv2.VideoCapture("Road traffic video for object recognition.mp4")

count = 0
tracker = Tracker()

cv2.namedWindow('Deteccion de vehiculos')
area1 = [(290, 325), (460, 330), (430, 445), (180, 390)]
area2 = [(570, 375), (735, 340), (870, 420), (580, 440)]
ir = set()
venir = set()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame = cv2.resize(frame, (1020, 600))
    results = model(frame)
    list = []
    for index, rows in results.pandas().xyxy[0].iterrows():
        x = int(rows[0])
        y = int(rows[1])
        x1 = int(rows[2])
        y1 = int(rows[3])
        b = str(rows['name'])
        list.append([x, y, x1, y1, b])
    idx_bbox = tracker.update(list)

    for bbox in idx_bbox:
        x2, y2, x3, y3, id, b = bbox
        cv2.rectangle(frame, (x2, y2), (x3, y3), (0, 0, 255), 2)
        cv2.putText(frame, str(id), (x2, y2), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
        cv2.circle(frame, (x3, y3), 3, (0, 255, 0), -1)

        results = cv2.pointPolygonTest(np.array(area1, np.int32), ((x3, y3)), False)
        results1 = cv2.pointPolygonTest(np.array(area2, np.int32), ((x3, y3)), False)

        if results > 0:
            ir.add((id,b,))
        if results1 > 0:
            venir.add((id, b,))
    print(ir)
    cv2.polylines(frame, [np.array(area1, np.int32)], True, (0, 255, 255), 3)
    cv2.polylines(frame, [np.array(area2, np.int32)], True, (0, 255, 255), 3)
    a1 = len(ir)
    a2 = len(venir)

    cv2.putText(frame, str(a1), (549, 465), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
    cv2.putText(frame, str(a2), (804, 411), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
    cv2.imshow("Deteccion de vehiculos", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
