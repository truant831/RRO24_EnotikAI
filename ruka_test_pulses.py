#import cv2 as cv
import numpy as np
import time
from libs.arm_control import set_position
#from video import camera
from libs.serial_pico import RPiPico_serial
pico = RPiPico_serial('/dev/ttyTHS1')


pos_card_center=(1995,1500, 1637) #100==1/3 stopki
pos_camera_home=(1995,1000, 2000)
pos_robots_card_down=(1300,1500, 1600)
pos_robots_card_up=(1300,1500, 2000)

# RGB STRIP COLORS = (BLACK, RED, YELLOW, GREEN, CYAN, BLUE, PURPLE, WHITE)

# LEFT 1900 1490 1430
# TOP LEFT 1453 1500 1456
# RIGHT 2156 1502 1458
# TOP RIGHT 2507 1500 1456

# range             action      #serv
# [970, 3020] - вращение        1
# [1000, 1500] - вперед - назад 2
# [1425, 2073] - вниз - вверх   3

# center of card zone =(1995,1500, 1490) #last card on table
# center of card zone =(1995,1500, 1600) #first card on table
# Robot card zone (1300,1500,1600)
# camera postion (1995,1000,2000)

for i in range(5):
    set_position(pos_camera_home)
    time.sleep(3)
    pico.apply(0,'GREEN')
    set_position((1995,1500,1800))
    time.sleep(0.7)
    set_position(pos_card_center)
    pos_card_center=tuple(np.subtract(pos_card_center, (0,0,3)))
    print(pos_card_center)
    time.sleep(1)
    pico.apply(1,'RED')
    time.sleep(1.5)
    pico.apply(1,'PURPLE')
    set_position((1995,1500,1800)) #need this not to move stopka cards away
    time.sleep(0.5)
    set_position(pos_robots_card_up)
    time.sleep(0.5)
    set_position(pos_robots_card_down)
    time.sleep(1)
    pico.apply(0,'BLACK')


# stream = camera()
# stream.start()

# calibration_param_path = "/home/eg/Documents/vladimir_ruka/calibration_param.npz"
# param_data = np.load(calibration_param_path)
# dim = tuple(param_data['dim_array'])
# k = np.array(param_data['k_array'].tolist())
# d = np.array(param_data['d_array'].tolist())
# scale = 1
# p = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(k, d, dim ,None)
# Knew = p.copy()
# if scale:
#     Knew[(0,1), (0,1)] = scale * Knew[(0,1), (0,1)]
# map1, map2 = cv.fisheye.initUndistortRectifyMap(k, d, np.eye(3), Knew, dim, cv.CV_16SC2)
# cap = 'v4l2src device=/dev/video0 io-mode=2 ! image/jpeg, width=(int)1280, height=(int)720, framerate=30/1 ! nvv4l2decoder mjpeg=1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink' 
# cam = cv.VideoCapture(cap, cv.CAP_GSTREAMER)
# while True:
#     __, frame = cam.read()
#     frame_calibrated = cv.remap(frame.copy(), map1, map2, interpolation=cv.INTER_CUBIC, borderMode=cv.BORDER_CONSTANT)
#     # frame = stream.read()
#     cv.imshow("camera",frame_calibrated)
#     if cv.waitKey(1) == ord('q'):
#         break

# demo

# set_position(1,1950)
# set_position(2,1100)
# set_position(3,2073)

# starting position
# for i in range(5):
  
# set_position(1,1900)
# set_position(2,1490)
# set_position(3,1470)
#     time.sleep(1.5)
#     pico.apply(1,'GREEN')
#     time.sleep(2)
#     set_position(3,2073)
#     time.sleep(1)
#     set_position(1,1453)
#     set_position(2,1500)
#     set_position(3,1456)
#     time.sleep(2)
#     pico.apply(0,'RED')

#     time.sleep(1.5)

#     set_position(1,2156)
#     set_position(2,1490)
#     set_position(3,1470)
#     time.sleep(1.5)
#     pico.apply(1,'GREEN')
#     time.sleep(2)
#     set_position(3,2073)
#     time.sleep(1)
#     set_position(1,2507)
#     set_position(2,1500)
#     set_position(3,1456)
#     time.sleep(2)
#     pico.apply(0,'RED')
#     set_position(1,1995)
#     set_position(2,1000)
#     set_position(3,1560)
#     time.sleep(1.5)

