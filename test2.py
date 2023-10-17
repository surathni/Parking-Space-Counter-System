import cv2
import numpy as np
import pickle
import pandas as pd
from ultralytics import YOLO
import cvzone
from datetime import datetime

# Loading previous data
with open("Parking3file", "rb") as f:
    data = pickle.load(f)
    polylines, parking_lots = data['polylines'], data['parking_lots']

# Loading YOLO model
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
model = YOLO('yolov8s.pt')

# Opening the video
cap = cv2.VideoCapture('Parking3.mp4')

count = 0

# Total Number of Parking Lots
total_parking_lots = len(polylines)



# Frame processing
while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    count += 1
    if count % 3 != 0:
        continue

    # Detecting the YOLO object
    frame = cv2.resize(frame, (1020, 500))
    frame_copy = frame.copy()
    results = model.predict(frame)

    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    list1 = []

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])

        c = class_list[d]
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        if 'car' in c:
            list1.append([cx, cy])

    counter1 = []
    list2 = []

    # Polygon and object intersection
    for i, polyline in enumerate(polylines):
        list2.append(i)
        cv2.polylines(frame, [polyline], True, (0, 255, 0), 2)
        cvzone.putTextRect(frame, f'{parking_lots[i]}', tuple(polyline[0]), 1, 1)
        for i1 in list1:
            cx1 = i1[0]
            cy1 = i1[1]
            result = cv2.pointPolygonTest(polyline, ((cx1, cy1)), False)
            if result >= 0:
                cv2.circle(frame, (cx1, cy1), 5, (255, 0, 0), -1)
                cv2.polylines(frame, [polyline], True, (0, 0, 255), 2)
                counter1.append(cx1)

    # Car and free space counting
    car_count = len(counter1)
    free_space = len(list2) - car_count

    # Calculate Occupancy Percentage
    occupancy_percentage = (car_count / total_parking_lots) * 100

    # Get current date and time
    current_datetime = datetime.now().strftime('%A, %Y-%m-%d %H:%M:%S')

    # Create a black frame for displaying information
    info_frame = np.zeros((200, 1020, 3), dtype=np.uint8)

    # Display Information
    cv2.putText(info_frame, f'Date and Time: {current_datetime}', (30, 30), 2, 1,(255, 255, 255))
    cv2.putText(info_frame, f'Total Number of Parking Lots: {total_parking_lots}', (30, 70), 2, 1,(255, 255, 255))
    cv2.putText(info_frame, f'Occupied Spaces: {car_count} ({occupancy_percentage:.2f}%)', (30, 110), 2, 1,(255, 255, 255))
    cv2.putText(info_frame, f'Free Spaces: {free_space} ({100 - occupancy_percentage:.2f}%)', (30, 150), 2, 1,(255, 255, 255))

    # Combine the video frame and the information frame
    combined_frame = np.vstack((frame, info_frame))

    # Display the combined frame
    cv2.imshow('FRAME', combined_frame)
    key = cv2.waitKey(1) & 0xFF

# Releasing resources
cap.release()
cv2.destroyAllWindows()
