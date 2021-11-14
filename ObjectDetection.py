import torch
import cv2 as cv
import numpy as np
from torch._C import Size
import matplotlib as plt

capture = cv.VideoCapture(0)

# Load Model
model = torch.hub.load('C:/Users/Iksan/Downloads/Compressed/yolov5-master', 'custom', path='path/to/Object_150.pt', source='local')  # local repo

while True:
    # Capture frame-by-frame
    ret, frame = capture.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Our operations on the frame come here
    results = model(cv.cvtColor(frame, cv.COLOR_BGR2RGB)) 
    
    # Calculate Multiple Object
    panjang = len(results.pandas().xyxy[0]) 
    converted = results.pandas().xyxy[0].to_numpy()
    # print(panjang)
    
    # Draw Bounding Box on Object Detected
    for i in range(panjang):
        cv.rectangle(frame, (int(converted[i][0]), int(converted[i][1])), (int(converted[i][2]), int(converted[i][3])), (255, 0, 0), 3)
        cv.putText(frame,f'{converted[i][6]} {int(converted[i][4]*100)}%',
                      (int(converted[i][0]), int(converted[i][1])), cv.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2) 
    
    # Display the resulting frame
    cv.imshow('frame', frame)
    
    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
capture.release()
cv.destroyAllWindows()