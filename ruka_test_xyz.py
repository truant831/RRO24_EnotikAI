
import sys
import os
import time
from libs.serial_pico import RPiPico_serial
pico = RPiPico_serial('/dev/ttyTHS1')
import numpy as np
from libs.arm_control_XYZ import *

# Default setting

SERVO_1_POS_LIMIT = [970, 3020]         # вращение 180deg
SERVO_2_POS_LIMIT = [1000, 1853]         # вперед - назад 75deg
SERVO_3_POS_LIMIT = [1365, 2048]         # вниз - вверх 60deg


# center of card zone =(1995,1500, 1490) #last card on table
# center of card zone =(1995,1500, 1600) #first card on table
# Robot card zone (1300,1500,1600)
# camera postion (1995,1000,2000)

L0 = 70 #84.4 # высота от стола до плоскости вращения основания
L1 = 8.14 # 
L2 = 128.4 # длина плеча левого мотора перпендикулярно вверх
L3 = 138.0 # длина от вершины прямого угла до оси end-of-tool
L4 = 16.8 # высота по Z инструмента ???


def test():
    rotate, left, right=(int((SERVO_1_POS_LIMIT[0]+SERVO_1_POS_LIMIT[1])/2),SERVO_2_POS_LIMIT [1],SERVO_3_POS_LIMIT[0]) #- стартовое положение 
    #rotate, left, right=(SERVO_1_POS_LIMIT[1],SERVO_2_POS_LIMIT [0],SERVO_3_POS_LIMIT[1]) #- стартовое положение 
    print("Pulses start ", (rotate, left, right))

    #переехало в pulse_to_deg in kinematics
    # #диапазон вращения сервы 1 - основание в градусах от 30 до 210 градусов, 120 это центр и по оси X=0
    # rotate_angle=val_map(rotate, SERVO_1_POS_LIMIT[0] , SERVO_1_POS_LIMIT[1], 30, 210)
    # #диапазон вращения сервы 2, отвечающей за вперед-назад, в градусах от 90 до 135 градусов
    # left_angle=val_map(left, SERVO_2_POS_LIMIT[0] , SERVO_2_POS_LIMIT[1], 90, 135)
    # #диапазон вращения сервы 3, отвечающей за вверх-вниз, в градусах от 50 до 0 градусов
    # right_angle=val_map(right, SERVO_3_POS_LIMIT[0] , SERVO_3_POS_LIMIT[1],50, 0)

    rotate_angle,left_angle, right_angle = pulse_to_deg((rotate,left,right))

    res_pos=list(forward_kinematics((rotate_angle,left_angle,right_angle)))

    for i in range(3):
        res_pos[i]=int(res_pos[i])
    print(" Coords start  ",res_pos)

    print("Test of inverse kinematics calc of angles ",inverse_kinematic((res_pos)))

    packetHandler.WritePosEx(1, rotate, SCS_MOVING_SPEED, SCS_MOVING_ACC) # вращение 
    packetHandler.WritePosEx(2, left, SCS_MOVING_SPEED, SCS_MOVING_ACC) # вперед-назад
    packetHandler.WritePosEx(3, right, SCS_MOVING_SPEED, SCS_MOVING_ACC) # вверх-вниз

#test()

set_position((0, (L1+L3+L4), L0+L2))
print(inverse_kinematic((0, (L1+L3+L4), L0+L2)))
pos_card_center=(1995,1500, 1637) #100==1/3 stopki
pos_camera_home=(1995,1000, 2000)
pos_robots_card_down=(1300,1500, 1600)
pos_robots_card_up=(1300,1500, 2000)


print("pos_card_center ", forward_kinematics(pulse_to_deg(pos_card_center)))
print("pos_camera_home ", forward_kinematics(pulse_to_deg(pos_camera_home)))
print("pos_robots_card_down ", forward_kinematics(pulse_to_deg(pos_robots_card_down)))
print("pos_robots_card_up ", forward_kinematics(pulse_to_deg(pos_robots_card_up)))

xyz_card_center=(0, 210, 25) #7
xyz_camera_home=(0,145, 180)
xyz_robots_card_down=(-210,0, 30)
xyz_human_card_down=(210,0,30)

set_position(xyz_camera_home)
time.sleep(3)

take_pos=(-55, 230, 25)
take_pos=(-55, 190, 25)
take_pos=(55, 190, 25)
take_pos=(55, 230, 25)

take_pos=xyz_card_center #центр всех карт

def move_card(position,adress):
    set_position(tuple(np.add(position, (0,0,20))))
    time.sleep(0.7)
    set_position(position)
    time.sleep(2.0)
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
    time.sleep(1.5)
    set_position(put_position)
    time.sleep(0.5)
    pico.apply(0,'BLACK')

for i in range(55):
    set_position(xyz_camera_home)
    time.sleep(1.5)
    pico.apply(0,'GREEN')
    move_card(xyz_card_center, "robot")
    set_position(xyz_card_center)
    xyz_card_center=tuple(np.subtract(xyz_card_center, (0,0,0.45))) #0.87
    print(i, xyz_card_center)

set_position(xyz_camera_home)