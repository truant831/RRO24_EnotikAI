# https://docs.ultralytics.com/ru/yolov5/tutorials/running_on_jetson_nano/#install-pytorch-and-torchvision
# https://jetsonhacks.com/2023/06/12/upgrade-python-on-jetson-nano-tutorial/

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import cv2
import numpy as np
import matplotlib.pyplot as plt
# importing os module   
import os 
import torch
import time

# Check for CUDA device and set it
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')


def morph_op(img, mode='open', ksize=5, iterations=1):
    im = img.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ksize, ksize))
     
    if mode == 'open':
        morphed = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)
    elif mode == 'close':
        morphed = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)
    elif mode == 'erode':
        morphed = cv2.erode(im, kernel)
    else:
        morphed = cv2.dilate(im, kernel)
     
    return morphed

def find_play_zone(image):
    #convert rgb to hsv
    blurred = cv2.GaussianBlur(image, (7, 7), 0)

    hsv_img = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    #cv2.imshow('Frame HSV', hsv_img)

    hsv_low = np.array([156, 43, 54], np.uint8)
    hsv_high = np.array([179, 255, 255], np.uint8)
    mask = cv2.inRange(hsv_img, hsv_low, hsv_high)
    #hue=165-180 and 0 to 30
    #sat=170 to 255
    #val=200 to 255
    hsv_low2 = np.array([165, 80, 80], np.uint8)
    hsv_high2 = np.array([180, 255, 255], np.uint8)
    mask2 = cv2.inRange(hsv_img, hsv_low2, hsv_high2)

    sum_mask=cv2.bitwise_or(mask,mask2)   
    dilated=morph_op(sum_mask, mode='dilate', ksize=13, iterations=2)
    morhped=morph_op(dilated, mode='open', ksize=5, iterations=2)
    contours, _ = cv2.findContours(morhped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    zone_box=()
    zone_image=None
    #cv2.imshow('Frame mask', morhped)

    if contours:
        # Сортируем контуры по убыванию площади
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Выбираем 1 самых больших контуров
        contour = sorted_contours[0]
        # Аппроксимируем контур
        approx = cv2.approxPolyDP(contour, 0.06 * cv2.arcLength(contour, True), True)
        #print(len(approx))

        if len(approx)==4:
            # Получим прямоугольник, описывающий контур
            x,y,w,h= cv2.boundingRect(contour)
            #print(zone_box)

            #заданы четыре ключевых точки маркера. Верх, низ, лево, право для оригинального квадрата
            new_coords=np.float32([[0,0],[0,h],[w,0],[w,h]])
            
            old_coords=approx.reshape(-1,2).astype(np.float32)
            old_coords=np.asarray(sorted(old_coords, key=lambda e:sum(e)))
            old_coords[1:3]=np.asarray(sorted(old_coords[1:3], key=lambda e:e[0]))

            #print(old_coords)
            #С помощью метода getPerspectiveTransform считается матрица трансформации, то, как трансформировались изображения
            M=cv2.getPerspectiveTransform(old_coords,new_coords)
            #методом warpPerspective убираем перспективу.
            zone_image=cv2.warpPerspective(image,M,(w,h))

    return zone_image

def get_card(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresholded = cv2.threshold(blurred, 140, 255, cv2.THRESH_BINARY) #180

    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    card_box=()
    cropped_image=None

    if contours:
        # Сортируем контуры по убыванию площади
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Выбираем 3 самых больших контуров
        largest_contours = sorted_contours[:1]  #3

        # Инициализируем переменные для хранения самого круглого контура и его круглости
        most_round_contour = None
        min_roundness = 1000

        # Перебираем выбранные контуры
        for contour in largest_contours:
            # Аппроксимируем контур
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Вычисляем круглость контура
            roundness = cv2.matchShapes(contour, approx, cv2.CONTOURS_MATCH_I2, 0.0)
            #   gives area of contour
            area = cv2.contourArea(contour)

            # Если круглость текущего контура лучше, чем у предыдущего самого круглого, обновляем значения
            if roundness < min_roundness and area>4000:
                min_roundness = roundness
                most_round_contour = contour
            #x, y, w, h = cv2.boundingRect(contour)
            #cv2.putText(image,str(roundness),(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)

        if most_round_contour is not None:
            # Создадим маску для контура
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [most_round_contour], 0, (255), thickness=cv2.FILLED)
            
            #decrease mask
            kernel=np.ones((15,15),np.uint8)
            mask=cv2.erode(mask,kernel)
            #cv2.imshow('Contours Frame', square_frame )
            # Применим маску к исходному изображению
            cropped_image= cv2.bitwise_and(image, image, mask=mask)
            # Получим прямоугольник, описывающий самый круглый контур
            x, y, w, h = cv2.boundingRect(most_round_contour)
            card_box=(x, y, w, h)
        
            # Обрежем изображение до размеров boundingRect
            cropped_image = cropped_image[y:y+h, x:x+w]

            #white all black and almost white
            cropped_image[np.where((cropped_image == [0,0,0]).all(axis = 2))]=[255,255,255]
            cropped_image[np.where((cropped_image > [155,155,155]).all(axis = 2))]=[255,255,255]
            
            '''
            # Increase contrast for each color channel using histogram equalization
            lab= cv2.cvtColor(cropped_image, cv2.COLOR_BGR2LAB)
            l_channel, a, b = cv2.split(lab)

            # Applying CLAHE to L-channel
            # feel free to try different values for the limit and grid size:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3,3))
            cl = clahe.apply(l_channel) 

            # merge the CLAHE enhanced L-channel with the a and b channel
            limg = cv2.merge((cl,a,b))

            # Converting image from LAB Color model to BGR color spcae
            enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            '''
            #return cropped_image, card_box #enhanced_img # 

    return cropped_image, card_box 



directory="/home/jetson/Documents/VSOH_TRAIN/"
os.chdir(directory)

model=YOLO('actual_model/best_1to8m.pt').to(device)

# Подключаем USB камеру
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
width = 1280 
height = 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# loading all the class labels (objects)
labels = model.names
print(labels)    

if not cap.isOpened():
    print("Cannot open camera")
    exit()
while cap.isOpened():
    # Чтение кадра
    
    ret, frame = cap.read()
    cv2.imshow('Video Frame', frame)

    key=cv2.waitKey(1) 
    #print (frame.shape)
    #print('Resolution: ' + str(frame.shape[0]) + ' x ' + str(frame.shape[1]))
    
    #debug color mask for zone
    #colormask(frame)
    #hue=90 to 150
    #sat=100 to 210
    #val=0 to 250
    
    zone_image=find_play_zone(frame)
    
    key=cv2.waitKey(1)

    if zone_image is not None:
        cv2.imshow('Play Zone', zone_image)

        # Find card and exctract it
        card, card_box  = get_card(zone_image)

        if card is not None:
            card=np.ascontiguousarray(card)
            cv2.imshow('Card', card)
            # YOLO predict model and show boxes
            results = model.predict(card)
            for r in results:
                annotator = Annotator(card)
                boxes = r.boxes
                for box in boxes:
                    b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
                    c = box.cls
                    annotator.box_label(b, labels[int(c)])         
                #boxes_xy = r.boxes.xyxy  # get box coordinates in (top, left, bottom, right) format
                #conf = r.boxes.conf   # confidence scores
                #cls = r.boxes.cls    # class labels 
            
            card_detected= annotator.result()  

            # Показ изображения с контурами
            cv2.imshow('Annotation', card_detected)

    
    # Остановка при нажатии клавиши 'q'
    if key & 0xFF == ord('q'):
        break

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()
