import cv2
from video import camera
import numpy as np

def nothing(x):
    pass

stream = camera()
stream.start()

name='HSV Filter'
cv2.namedWindow(name,1)

#set trackbar
hh = 'hue high'
hl = 'hue low'
sh = 'saturation high'
sl = 'saturation low'
vh = 'value high'
vl = 'value low'
mode = 'mode'

#set ranges
cv2.createTrackbar(hh, name, 0,179, nothing)
cv2.createTrackbar(hl, name, 0,179, nothing)
cv2.createTrackbar(sh, name, 0,255, nothing)
cv2.createTrackbar(sl, name, 0,255, nothing)
cv2.createTrackbar(vh, name, 0,255, nothing)
cv2.createTrackbar(vl, name, 0,255, nothing)
cv2.createTrackbar(mode, name, 0,3, nothing)
thv= 'th1'
cv2.createTrackbar(thv, name, 127,255, nothing)

while True:
    # Чтение кадра
    
    frame = stream.read()
    #cv2.imshow("Original", frame)
    key=cv2.waitKey(1) 
    #print (frame.shape)
    #print('Resolution: ' + str(frame.shape[0]) + ' x ' + str(frame.shape[1]))
    
    # convert img to grayscale
    imgg = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #convert rgb to hsv
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    hul= cv2.getTrackbarPos(hl,name)
    huh= cv2.getTrackbarPos(hh,name)
    sal= cv2.getTrackbarPos(sl,name)
    sah= cv2.getTrackbarPos(sh,name)
    val= cv2.getTrackbarPos(vl,name)
    vah= cv2.getTrackbarPos(vh,name)
    thva= cv2.getTrackbarPos(thv,name)

    modev= cv2.getTrackbarPos(mode,name)

    hsvl = np.array([hul, sal, val], np.uint8)
    hsvh = np.array([huh, sah, vah], np.uint8)

    mask = cv2.inRange(hsv_img, hsvl, hsvh)

    res = cv2.bitwise_and(frame, frame, mask=mask)

    #set image for differnt modes
    ret, threshold = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    ret, img_th= cv2.threshold(imgg, thva, 255, cv2.THRESH_TOZERO)
    res2 = cv2.bitwise_and(img_th, img_th, mask=threshold)
    res_rgb = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
    #convert black to white
    res[np.where((res==[0,0,0]).all(axis=2))] = [255,255,255]

    if modev ==0:
        #show mask only
        cv2.imshow(name,mask)
    elif modev ==1:
        #show white-masked color img
        cv2.imshow(name,res)
    elif modev ==2:
        #show white-masked binary img with threshold
        cv2.imshow(name,threshold)
    else:
        #white-masked grayscale img with threshold
        cv2.imshow(name,res2)

    # Остановка при нажатии клавиши 'q'
    if key & 0xFF == ord('q'):
        break
    if key & 0xFF == ord('s'):
        cv2.imwrite("camera_frame.jpg", frame)


# Освобождаем ресурсы
stream.stop()
cv2.destroyAllWindows()