import cv2
from libs.video import camera
from libs.play_sounds import Say_card_name
from libs.video_predict import predict_card, compare_cards, extract_classes, draw_match_class

import time
import os 
import numpy as np
#best1to8_newS.pt

directory="/home/jetson/Documents/VSOSH_region/"
os.chdir(directory)

stream = camera()
stream.start()

usl=False
time_last_predict=time.time()
n_detected=0
my_card_classes=np.zeros(1)
score_robot=0

while not usl:
    frame = stream.read()
    cv2.imshow('Video Frame', frame)

    #do not delete waitkey, will not work
    if cv2.waitKey(1) == ord('q'):
       break
    
    #запускаем распознавание не чаще чем раз в 1 сек
    if (time.time()-time_last_predict)>1:
        img_card_detected, play_card_results, card_box = predict_card(frame)
        time_last_predict=time.time()
    else:
        img_card_detected=None

    if img_card_detected is not None:
        cv2.imshow('Play card', img_card_detected)
        play_card_classes, n_detected, boxes =extract_classes(play_card_results,"Play zone card")
        (x,y,w,h)=card_box
        print("Card center " +str(int(x+w/2))+" , "+str(int(y+h/2)))
        if len(my_card_classes)==1:
            my_card_classes=play_card_classes

    #выйти из цикла если нашли 8 объектов на карте
    if n_detected==8:
    #    usl=True
        #набор классов изменился, значит другая карта перед нами
        if not np.array_equal(play_card_classes, my_card_classes):
            dobble_name=compare_cards(my_card_classes, play_card_classes) # сравнить свою карту и карту поля
            # запоминаем свою карту
            img_my_card=img_card_detected
            my_card_results=play_card_results
            my_card_classes=play_card_classes
            if len(dobble_name) > 0 and n_detected==8:
                #добавим себе балл
                score_robot+=1                
                #создаем копии картинок с обведенной рамкой обнаруженного объекта
                img_my_card_show=draw_match_class(img_my_card,my_card_results,dobble_name[0])
                img_play_card_show=draw_match_class(img_card_detected,play_card_results,dobble_name[0])
                #можно вот так в одну объединить картинку и потом ее показать, если не под одной 
                #Hori = np.concatenate((img_my_card_show, img_play_card_show), axis=1) 
                #cv2.imshow('Match!', Hori)    

                #показываем по одной
                cv2.imshow('My_card', img_my_card_show)
                cv2.imshow('Play card', img_play_card_show)
        
                # сказать что нашел
                Say_card_name(dobble_name[0])
                # wait button
                key=cv2.waitKey(1)

stream.stop()
cv2.destroyAllWindows()