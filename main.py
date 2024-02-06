import cv2
import numpy as np
import torch
from tracker import *
from datetime import datetime
import csv

# Cargar el modelo YOLOv5 preentrenado
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Capturar video desde un archivo
cap = cv2.VideoCapture("Road traffic video for object recognition.mp4")
ir_csv = 'ir.csv'
count = 0
tracker = Tracker()

# Crear una ventana para mostrar la detección de vehículos
cv2.namedWindow('Deteccion de vehiculos')

# Definir áreas de interés y configurar objetos para el seguimiento
area1 = [(290, 325), (465, 325), (465, 335), (290, 335)]
area2 = [(545, 340), (775, 340), (775, 355), (545, 355)]
ir = set()
venir = set()
datos_ir = ()

# Configurar la frecuencia de cuadros objetivo y tiempo de espera
fps_target = 24
wait_time = int(1000 / fps_target)

# Bucle principal para procesar cada cuadro del video
while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    # Redimensionar el cuadro para ajustarlo al modelo YOLOv5 y realizar la detección de vehiculos
    frame = cv2.resize(frame, (1020, 600))
    results = model(frame)
    # Obtener las coordenadas de los cuadros delimitadores y las etiquetas de los objetos detectados
    bbox_list = []
    for index, rows in results.pandas().xyxy[0].iterrows():
        x = int(rows[0])
        y = int(rows[1])
        x1 = int(rows[2])
        y1 = int(rows[3])
        b = str(rows['name'])
        bbox_list.append([x, y, x1, y1, b])

    # Actualizar el rastreador de objetos y obtener los resultados del seguimiento llamando a la clase tracker
    idx_bbox = tracker.update(bbox_list)


    # Dibujar los cuadros delimitadores y realizar acciones según las áreas de interés
    for bbox in idx_bbox:
        x2, y2, x3, y3, id, b = bbox
        cv2.rectangle(frame, (x2, y2), (x3, y3), (0, 0, 255), 2)
        cv2.putText(frame, str(id), (x2, y2), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
        cv2.circle(frame, (x3, y3), 3, (0, 255, 0), -1)

        # Comprobar si los objetos están dentro de las áreas de interés y registrar eventos
        results = cv2.pointPolygonTest(np.array(area1, np.int32), ((x3, y3)), False)
        results1 = cv2.pointPolygonTest(np.array(area2, np.int32), ((x3, y3)), False)

        #recoge la fecha
        fecha_actual = datetime.now()

        #guarda los datos de los vehiculos en el CSV y en la coleccion
        if results > 0:
            datos_ir = (id, b, fecha_actual.strftime("%Y-%m-%d"), fecha_actual.strftime("%H:%M"))
            if datos_ir not in ir:
                with open('ir.csv', 'a', newline='') as archivo_csv:
                    escritor = csv.writer(archivo_csv)
                    escritor.writerow(datos_ir)
                ir.add(datos_ir)

        if results1 > 0:
            datos_venir = (id, b, fecha_actual.strftime("%Y-%m-%d"), fecha_actual.strftime("%H:%M"))
            if datos_venir not in venir:
                print(datos_venir)
                with open('venir.csv', 'a', newline='') as archivo_csv:
                    escritor = csv.writer(archivo_csv)
                    escritor.writerow(datos_venir)
                venir.add(datos_venir)

    # Dibujar áreas de interés y mostrar estadísticas en la ventana
    cv2.polylines(frame, [np.array(area1, np.int32)], True, (0, 255, 255), 3)
    cv2.polylines(frame, [np.array(area2, np.int32)], True, (0, 255, 255), 3)

    a1 = len(ir)
    a2 = len(venir)

    cv2.putText(frame, str(a1), (435, 375), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
    cv2.putText(frame, str(a2), (804, 431), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
    cv2.imshow("Deteccion de vehiculos", frame)

    # Romper el bucle si se presiona la tecla 'Esc'
    if cv2.waitKey(wait_time) & 0xFF == 27:
        break

# Liberar recursos y cerrar la ventana
cap.release()
cv2.destroyAllWindows()
