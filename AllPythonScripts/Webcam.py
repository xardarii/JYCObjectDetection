# Importing opencv for computer vision
import cv2
# Importing matplotlib so we can visualize an image
from matplotlib import pyplot as plt
import numpy as np
import torch
import uuid
import os
import time

model = torch.hub.load('YOLORepository/yolov7','custom',path= 'F:/YOLORepository/yolov7/runs/train/exp/weights/best.pt', force_reload=True)

#Connect to webcam
cap = cv2.VideoCapture(0)
#Loop through every frame until we close it
while cap.isOpened():
    ret, frame = cap.read()
    
    #Make detection
    results = model(frame)

    cv2.imshow('Webcam',np.squeeze(results.render()))
    # Checks whether Q has been hit and stops the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
