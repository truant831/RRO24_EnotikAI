import sys
import cv2
import torch
from ultralytics import YOLO

print("Python Version "+sys.version)

print("Open CV version "+cv2.__version__)

count = cv2.cuda.getCudaEnabledDeviceCount()
print("Number of CUDA devices for OpenCV "+str(count))

print("Torch with cuda:" + str(torch.cuda.is_available()))

import serial
ser = serial.Serial('/dev/ttyTHS1')
print("Serila ttyTHS1 was opened")
