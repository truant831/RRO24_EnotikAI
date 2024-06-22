
import time
import os 

directory="/home/jetson/Documents/VSOSH_final/"
os.chdir(directory)

from libs.arm_control_XYZ import set_position
from libs.serial_pico import RPiPico_serial
pico = RPiPico_serial('/dev/ttyTHS1')
from libs.play_sounds import Say_card_name
from libs.video_predict import find_play_zone, get_card, extract_classes
from libs.video import camera

#https://pythonist.ru/kak-importirovat-v-python/


# propisat v comand line: cd /dev/; sudo chmod 666 ttyTHS1;sudo chmod 666 ttyTHS2;

import cv2
import torch
import numpy as np

from ultralytics import YOLO
# Check for CUDA device and set it
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

model=YOLO('actual_model/best1to8_newS.pt/').to(device)
#loading all the class labels (objects)
labels = model.names

counter=6
prefix="doble_train_img_"

pos_camera_home=(0,160, 180)

def save_annotations(img, bboxes, class_labels):
    global prefix
    global counter
    global directory

    os.chdir(directory+"pictures_new/") 
    # Saving the image 
    filename=prefix+"_"+str(counter)
    cv2.imwrite(filename+".jpg", img)

    img_height = img.shape[0]
    img_width = img.shape[1]
 
    with open(filename+".txt", 'w') as f:
        for i, box in enumerate(bboxes):
            x1, y1 = box[0], box[1]
            x2, y2 = box[2], box[3]
             
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
                 
            width = x2 - x1
            height = y2 - y1
            x_centre, y_centre = int(width/2)+x1, int(height/2)+y1
 
            norm_xc = x_centre/img_width
            norm_yc = y_centre/img_height
            norm_width = width/img_width
            norm_height = height/img_height

            #print(i)
            #print(class_labels[i])
            yolo_annotations = [str(class_labels[i]), ' ' + str(norm_xc), 
                                ' ' + str(norm_yc), 
                                ' ' + str(norm_width), 
                                ' ' + str(norm_height), '\n']
             
            f.writelines(yolo_annotations)
    counter+=1
    print("sucsessfull")
    time.sleep(0.5)

# Подключаем USB камеру
stream = camera()
stream.start()

set_position(pos_camera_home) # приезжаем на фиолетовый фон
usl=False
time_last_predict=time.time()
n_detected=0

while not usl:
    frame = stream.read()
    cv2.imshow('Video Frame', frame)

    key=cv2.waitKey(1)
    #do not delete waitkey, will not work
    if key== ord('q'):
       break
    
    #запускаем распознавание не чаще чем раз в 0.5 сек
    if (time.time()-time_last_predict)>0.5:
        zone_image=find_play_zone(frame)
        if zone_image is not None:
            #cv2.imshow('Play Zone', zone_image)
            # Find card and exctract it
            card, card_box  = get_card(zone_image)
            if card is not None:
                card=np.ascontiguousarray(card)
                
                card_detected=card.copy()
                #YOLO predict model and show boxes
                results_card = model.predict(card)[0].boxes
                n_objects=0
                for box in results_card:
                    left, top, right, bottom = np.array(box.xyxy.cpu(), dtype=int).squeeze()
                    width = right - left
                    height = bottom - top
                    center = (left + int(width/2), top + int(height/2))
                    label = labels[int(box.cls.cpu())] #results[0].names[int(box.cls)]
                    confidence = float(box.conf.cpu())
                    confidence = int(float(box.conf.cpu())*100)
                    
                    cv2.rectangle(card_detected, (left, top),(right, bottom), (80, 80, 80), 2)
                    cv2.putText(card_detected, label,(left, top-10),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.putText(card_detected, str(confidence),(left, top+20),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
                # Показ изображения с контурами
                cv2.imshow('Annotation', card_detected)
                cv2.imshow('Card', card)
                key=cv2.waitKey(50)
                play_card_classes, n_detected, b_boxes=extract_classes(results_card,"Train card")
            time_last_predict=time.time()
    else:
        card_detected=None
    
    if n_detected==10 or key & 0xFF == ord('s'):
        print("Are you sure to save labels for card?")
        key=cv2.waitKey(0)
        if key & 0xFF == ord('w'):
            print(b_boxes)
            print(play_card_classes)
            save_annotations(card,b_boxes,play_card_classes)

# Освобождаем ресурсы
stream.stop()
cv2.destroyAllWindows()
pico.apply(0,'BLACK')   