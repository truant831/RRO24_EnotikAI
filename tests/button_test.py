import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BOARD)
but_pin = 15
GPIO.setup(but_pin, GPIO.IN)

while 1==1:
    print(not GPIO.input(but_pin))

GPIO.cleanup()             