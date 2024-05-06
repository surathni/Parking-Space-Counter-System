#Importing relevant libraries (cv2-computer vision task/ cvzone-contain utility functions for drawing text on frames
# numpy-for numerical and array operations/ pickle-serializing and deserializing objects)
import cv2
import cvzone
import numpy as np
import pickle


#Opening the video file
cap = cv2.VideoCapture('Parking4.mp4')


#Initializing variables
drawing=False
parking_lots=[]


#Loading previous data from the files
try:
    with open("Parking4file", "rb") as f:
          data=pickle.load(f)
          polylines, parking_lots= data['polylines'], data['parking_lots']
except:
    polylines=[]


#Initializing variables
points=[]
current_name=" "



# Event handling - mouse callback
def draw(event, x, y, flags, param):
    global points, drawing
    drawing = True
    if event == cv2.EVENT_LBUTTONDOWN:
        points = [(x, y)]
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            points.append((x, y))
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        current_name = input('Parking Lot:')
        if current_name:
            parking_lots.append(current_name)
            # Find the minimum and maximum coordinates to create a bounding rectangle
            min_x = min(points, key=lambda p: p[0])[0]
            max_x = max(points, key=lambda p: p[0])[0]
            min_y = min(points, key=lambda p: p[1])[1]
            max_y = max(points, key=lambda p: p[1])[1]
            # Append the rectangle as four points (top-left, top-right, bottom-right, bottom-left)
            polylines.append(np.array([(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)], np.int32))


# Video frame processing loop
while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    frame = cv2.resize(frame, (680, 500))

    for i, polyline in enumerate(polylines):
        # Draw rectangles on the frame and add text labels
        rect = cv2.boundingRect(polyline)
        cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), 2)
        cvzone.putTextRect(frame, f'{parking_lots[i]}', (rect[0], rect[1]), 1, 1)


#Display the frame
    cv2.imshow('FRAME', frame)
    #Enable drawing on the video frame
    cv2.setMouseCallback('FRAME', draw)

    Key = cv2.waitKey(100) & 0xFF
    #Saving the data
    if Key==ord('s'):
        with open("Parking4file", "wb") as f:
            data={'polylines':polylines,'parking_lots':parking_lots}
            pickle.dump(data,f)


#After the loop ends, code closes the video capture and close all opencv windows
cap.release()
cv2.destroyAllWindows()