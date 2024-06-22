
import __main__
from libs.button import *
from libs.play_sounds import Say_phraze
import RPi.GPIO as GPIO
import time

import http.client as httplib

def checkInternetHttplib(url="tts.api.cloud.yandex.net", timeout=3):
    connection = httplib.HTTPConnection(url, timeout=timeout)
    try:
        print("Checking Internet connection")
        # only header requested for fast operation
        connection.request("HEAD", "/")
        connection.close()  # connection closed
        print("Internet On")
        return True
    except Exception as exep:
        print(exep)
        return False
is_online=checkInternetHttplib()

if is_online:
    from libs.YaSpeech import generate_speech,recognize_speech

P_BUTTON = 15 # adapt to your wiring
is_pressed=False

def setup():
    GPIO.setmode(GPIO.BOARD)
    button = Button(P_BUTTON) 
    button.addXButtonListener(onButtonEvent)

def onButtonEvent(button, event):
    global is_pressed
    if event == BUTTON_PRESSED:
        print ("pressed")
        is_pressed=True
    elif event == BUTTON_RELEASED:
        print ("released") 
        is_pressed=False    
    elif event == BUTTON_LONGPRESSED:
       print ("long pressed")
    elif event == BUTTON_CLICKED:
        print ("clicked")
        if not __main__.match_found: #не засчитываем щелчок если сейчас роботт в состоянии "нашел пару
            __main__.was_clicked=True
    elif event == BUTTON_DOUBLECLICKED:
        if is_online:
            try:
                __main__.isSpeaking=True
                print ("double clicked")
                print("Dialog start")
                #phraza="Слушаю Вас"
                Say_phraze("listen")
                #generate_speech(phraza)
                text=recognize_speech()
                text=text.lower()
                print("Пользователь сказал: "+text)
                if ("счет" in text) or ("счёт" in text):
                    if __main__.score_robot>__main__.score_human:
                        phraza="Сейчас счёт "+str(__main__.score_robot)+" : "+str(__main__.score_human)+" в мою пользу"
                    elif __main__.score_robot==__main__.score_human:
                        phraza="Сейчас ничья "+str(__main__.score_robot)+" : "+str(__main__.score_human)
                    else:
                        phraza="Сейчас счёт "+str(__main__.score_human)+" : "+str(__main__.score_robot)+" в твою пользу"
                    generate_speech(phraza)
                if ("время" in text) or (" час" in text):
                    # Local time has date and time
                    t = time.localtime()
                    # Extract the time part
                    #current_time = time.strftime("%H:%M:%S", t)
                    hour_minutes=time.strftime("%H:%M",t)
                    #minutes=time.strftime("%M",t)
                    phraza="текущее время "+hour_minutes
                    generate_speech(phraza)
                if ("кто" in text) and (("автор" in text) or ("сделал" in text)):
                    phraza="Автор этого проекта Егор Каржавин, ученик восьмого А класса Лицея Иннополис"
                    generate_speech(phraza)
            except:
                print("Yandex speech error")
            __main__.isSpeaking=False
       

def clean_btn():
    GPIO.cleanup()

def wait_time_or_btn(timeout=3):
    global is_pressed
    start_time=time.time()
    #is_pressed=not GPIO.input(P_BUTTON )

    while not is_pressed:
        #is_pressed=not GPIO.input(P_BUTTON)
        elapsed_time=time.time()-start_time

        if elapsed_time>timeout:
            print("Button wait time is over. was not pressed")
            break
        time.sleep(0.1)
    return is_pressed

setup()