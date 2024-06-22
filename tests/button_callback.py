# Button3.py

from button import *
import RPi.GPIO as GPIO
import time

P_BUTTON = 15 # adapt to your wiring

def setup():
    GPIO.setmode(GPIO.BOARD)
    button = Button(P_BUTTON) 
    button.addXButtonListener(onButtonEvent)

def onButtonEvent(button, event):
    global isRunning
    if event == BUTTON_PRESSED:
        print ("pressed")
    elif event == BUTTON_RELEASED:
        print ("released")
    elif event == BUTTON_LONGPRESSED:
       print ("long pressed")
    elif event == BUTTON_CLICKED:
        print ("clicked")
    elif event == BUTTON_DOUBLECLICKED:
        print ("double clicked")
        isRunning = False
       
setup()
isRunning = True
while isRunning: 
    time.sleep(0.1)

GPIO.cleanup()
print ("all done")