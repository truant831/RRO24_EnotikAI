import time
import os 
directory="/home/jetson/Documents/VSOSH_final/"
os.chdir(directory)

from libs.YaSpeech import generate_speech
from libs.arm_control_XYZ import set_position
from libs.serial_pico import RPiPico_serial
pico = RPiPico_serial('/dev/ttyTHS1')
from libs.play_sounds import Say_card_name, Say_phraze
from libs.video_predict import predict_card, compare_cards, extract_classes, draw_match_class
from libs.button_callback import clean_btn, wait_time_or_btn, is_online
from libs.video import camera

#https://pythonist.ru/kak-importirovat-v-python/

# propisat v comand line: cd /dev/; sudo chmod 666 ttyTHS1;sudo chmod 666 ttyTHS2;

import cv2
import torch
import numpy as np


pos_card_center=(0, 210, 23.5)  
pos_camera_home=(0,160, 180)

xyz_robots_card_down=(-210,0, 10)
xyz_human_card_down=(210,0,30)

pos_camera_home_start=(-150, 0, 180) #старт для просмотра карты на голубом, подобрать
pos_camera_home_fuman=(150,0,180) #подобрать

isSpeaking=False
match_found=False
debug_mode=False

def move_card(position,adress):
    set_position(tuple(np.add(position, (0,0,20))))
    time.sleep(0.6)
    set_position(position)
    time.sleep(0.8)
    pico.apply(1,'RED')
    time.sleep(1)
    pico.apply(1,'PURPLE')
    set_position(tuple(np.add(position, (0,0,20))))
    time.sleep(0.5)
    if adress=="robot":
        put_position=xyz_robots_card_down
    else:
        put_position=xyz_human_card_down
    set_position(tuple(np.add(put_position, (0,0,60))))
    time.sleep(0.5)
    match_found=False
    was_clicked=False #Обнуляем нажатия человека, так как карту уже отнесли на полпути текущую, а новую он не мог успеть увидеть
    time.sleep(1.0)
    set_position(put_position)
    time.sleep(0.5)
    pico.apply(0,'BLACK')

# RGB STRIP COLORS = (BLACK, RED, YELLOW, GREEN, CYAN, BLUE, PURPLE, WHITE)  #какие есть цвета и как писать

# Подключаем USB камеру

stream = camera()
stream.start()

set_position(pos_camera_home_start) # приезжаем на синий фон чтоб считать свою карту

#wait for a button 1 sec , if not start
pico.apply(0,'GREEN')
wait_time_or_btn(1)
pico.apply(0,'BLACK')

# находим свою карту и 
usl=False
time_last_predict=time.time()
n_detected=0

while not usl:
    frame = stream.read()
    cv2.imshow('Video Frame', frame)

    #do not delete waitkey, will not work
    if cv2.waitKey(20) == ord('q'):
       break
    
    #запускаем распознавание не чаще чем раз в 2 сек
    if (time.time()-time_last_predict)>1:
        img_card_detected, play_card_results, card_box, zone_box = predict_card(frame)
        time_last_predict=time.time()
    else:
        img_card_detected=None

    if img_card_detected is not None:
        #cv2.imshow('Annotation', img_card_detected)
        play_card_classes, n_detected, boxes =extract_classes(play_card_results,"Play zone card")
        a=0
        
    #выйти из цикла если нашли 8 объектов на карте
    if n_detected==8:
        usl=True
    #как вариант сделать таймер чтоб если больше 10 секунд не может найти то просить фумана помоч и повернуть карту

# запоминаем свою карту
img_my_card=img_card_detected
my_card_results=play_card_results
my_card_classes=play_card_classes

cv2.imshow('My_card', img_my_card)
img_play_card=np.ones_like(img_my_card)
cv2.imshow('Play_card', img_play_card)
cv2.waitKey(100) 

#поехать на центральную зону
set_position(pos_camera_home)

#либо поморгать 5 раз и подождать, либо кнопку нажали и поедет
count=0
while count<5:
    pico.apply(0,'PURPLE')
    btn=wait_time_or_btn(0.5)
    pico.apply(0,'BLACK')
    btn=wait_time_or_btn(0.5)
    count+=1
    if btn:
        break

usl=False
time_last_predict=time.time()
score_robot=0
score_human=0
t_pause=0

#попробуем найти карту и распознать ее, надо на случай если человек нажмет раньше робота первую карту в колоде, чтобы робот знал где брать ему карту
img_card_detected, play_card_results, card_box, zone_box = predict_card(frame)
(x,y,w,h)=card_box
card_x=int(x+w/2)
card_y=int(y+h/2)
print("Card center " +str(card_x)+" , "+str(card_y))
was_clicked=False
key=cv2.waitKey(1)

while not usl: 
    if was_clicked and card_x>0 and card_y>0: #обновляется через отдельный поток и callback 
        # перенос карты фумену
        #добавим балл человеку
        score_human+=1 
        pos_x, pos_y, pos_z= pos_card_center
        pos_x=int(0+(card_x-340)/4) #recalc x where we will take card based on card center on camera
        pos_y=int(210-(card_y-250)/4) #recalc y where we will take card based on card center on camera
        pos_z=pos_z-0.45
        pos_card_center=(pos_x, pos_y, pos_z)
        was_clicked=False
        move_card(pos_card_center,"human")
        pos_card_center=tuple(np.subtract(pos_card_center, (0,0,0.45)))
        set_position(pos_camera_home)
        time.sleep(1)
    else:
        match_found=False
        #обнуляем щелчок кнопки, чтобы потом смотреть был ли он за время одной итерации цикла
        was_clicked=False
        # Чтение кадра
        frame = stream.read()
        if not debug_mode:
            cv2.imshow('Video Frame', frame)
            key=cv2.waitKey(1) #без этог плохо обновлят кадр в окне просмотра

        #запускаем распознавание не чаще чем раз в 0.5+t_pause сек
        if (time.time()-time_last_predict)>(0.5+t_pause):
            img_card_detected, play_card_results, card_box, zone_box = predict_card(frame)
            time_last_predict=time.time()
            t_pause=0
        else:
            img_card_detected=None

        if img_card_detected is not None:
            #координаты центра карты в координатах камеры, относительно Кадра с прямоугольником игровой зоны
            (x,y,w,h)=card_box
            card_x=int(x+w/2)
            card_y=int(y+h/2)
            print("Card center " +str(card_x)+" , "+str(card_y))
            #считам координаты центра карты в кординатах реального мира для руки
            pos_x, pos_y, pos_z= pos_card_center
            pos_x=int(0+(card_x-340)/4) #recalc x where we will take card based on card center on camera
            pos_y=int(210-(card_y-250)/4) #recalc y where we will take card based on card center on camera

            if debug_mode:
                print("Adding debug info on card image")
                cv2.circle(img_card_detected, (int(w/2), int(h/2)),10, (100, 255, 0), -1) #круг в центре (по координатам камеры внутри карты) с заливкой (толщина=-1)
                cv2.putText(img_card_detected, str(pos_card_center[0])+";"+str(pos_card_center[1]),(int(w/2) -40, int(h/2)+40),cv2.FONT_HERSHEY_COMPLEX, 0.9, (100, 255, 0), 1, cv2.LINE_AA) #напишем координаты реального мира
                #main frame
                zone_x=zone_box[0]
                zone_y=zone_box[1]
                cv2.circle(frame, (zone_x+card_x, zone_y+card_y),10, (100, 255, 0), -1) #круг в центре (по координатам камеры внутри карты) с заливкой (толщина=-1)
                cv2.imshow('Video Frame', frame)

            cv2.imshow('Play_card', img_card_detected)
            key=cv2.waitKey(20)

            #для отладки можно комментировать строку ниже и смотреть как обновляется вообще PLay_card
            play_card_classes, n_detected, boxes =extract_classes(play_card_results,"Play zone card")

            #проверить совпадения в картинках своей карты и карты поля
            dobble_name=compare_cards(my_card_classes, play_card_classes) # сравнить свою карту и карту поля
            
            # перенос карты себе если нашли совпадение и человек не успел
            if len(dobble_name) > 0 and n_detected==8 and was_clicked==False: 
                match_found=True
                pos_z=pos_z-0.45 #вычтем Z чтобы в след раз брать карту чуть ниже
                pos_card_center=(pos_x, pos_y, pos_z)
                pico.apply(0,'GREEN')
                #добавим себе балл
                score_robot+=1                
                #создаем копии картинок с обведенной рамкой обнаруженного объекта
                img_my_card_show=draw_match_class(img_my_card,my_card_results,dobble_name[0])
                img_play_card_show=draw_match_class(img_card_detected,play_card_results,dobble_name[0])
                #можно вот так в одну объединить картинку и потом ее показать, если не под одной, но надо еще их одиноковой высоты сделать перед объединением
                #Hori = np.concatenate((img_my_card_show, img_play_card_show), axis=1) 
                #cv2.imshow('Match!', Hori)    

                #показываем по одной
                cv2.imshow('My_card', img_my_card_show)
                cv2.imshow('Play_card', img_play_card_show)
                key=cv2.waitKey(50)

                #подождать пока говорит в другом потоке
                while isSpeaking:
                    time.sleep(0.1)
                # сказать что нашел
                Say_card_name(dobble_name[0])

                # перенос карты себе
                move_card(pos_card_center,"robot")
                pos_robots_card_down=tuple(np.add(xyz_robots_card_down, (0,0,0.45)))
                set_position(pos_camera_home)
                time.sleep(1)
                
                # запоминаем свою карту
                img_my_card=img_card_detected
                my_card_results=play_card_results
                my_card_classes=play_card_classes
                cv2.imshow('My_card', img_my_card)
                img_play_card=np.ones_like(img_play_card_show)
                cv2.imshow('Play_card', img_play_card)
                cv2.waitKey(20)
                time_last_predict=time.time()
                t_pause=2        
    
    # Выйдем из программы при нажатии клавиши 'q'
    if key & 0xFF == ord('q'):
        break
    # переключить режим отладки
    if key & 0xFF == ord('d'):
        debug_mode=True
        print("Debugging ", debug_mode)
    # переключить режим отладки
    if key & 0xFF == ord('a'):
        debug_mode=False
        print("Debugging ", debug_mode)

    #условие что закончились карты
    if (score_robot+score_human)>5: #55-2=53 for full set, 5 for test prog
        usl=True
        print("Robot", score_robot)
        print("Human", score_human)
    # img_card_detected - картинка с аннотоциями
    # play_card_results - результаты Yolo игровой карты
    # play_card_classes - numpy массив с номерами классов карты на поле

#говорим прощальные фразы
if is_online:
    if score_robot>score_human:
        phraza="игра закончилась со счётом "+str(score_robot)+" : "+str(score_human)+" в мою пользу"
    elif score_robot==score_human:
        phraza="игра закончилась в ничью "+str(score_robot)+" : "+str(score_human)
    else:
        phraza="игра закончилась со счётом "+str(score_human)+" : "+str(score_robot)+" в твою пользу"
    generate_speech(phraza)

if score_robot>score_human:
    Say_phraze("robot_win")
    print("robot_win")
elif score_robot==score_human:
    Say_phraze("nobody")
    print("nobody")
else:
    Say_phraze("human_win")
    print("human_win")

# Освобождаем ресурсы
stream.stop()
cv2.destroyAllWindows()
clean_btn()
pico.apply(0,'BLACK')   
