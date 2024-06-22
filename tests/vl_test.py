import time
from machine import Pin, I2C
from vl53l0x import VL53L0X

print("setting up i2c")
sda = Pin(6)
scl = Pin(7)
id = 1

i2c = I2C(id=id, sda=sda, scl=scl)

print(i2c.scan())

print("creating vl53lox object")
# Create a VL53L0X object
tof = VL53L0X(i2c)

# the measuting_timing_budget is a value in micro seconds, the
# longer the budget, the more accurate the reading. (originally 40000)
budget = tof.measurement_timing_budget_us
print("Budget was:", budget)
tof.set_measurement_timing_budget(100000)

# Sets the VCSEL (vertical cavity surface emitting laser) pulse period
# for the given period type (VL53L0X::VcselPeriodPreRange or
# VL53L0X::VcselPeriodFinalRange) to the given value (in PCLKs).
# Longer periods increase the potential range of the sensor. 
# Valid values are (even numbers only):

# tof.set_Vcsel_pulse_period(tof.vcsel_period_type[0], 18) 12 default
tof.set_Vcsel_pulse_period(tof.vcsel_period_type[0], 18)

# tof.set_Vcsel_pulse_period(tof.vcsel_period_type[1], 14) 8 default
tof.set_Vcsel_pulse_period(tof.vcsel_period_type[1], 14)

# Number of readings to average
n = 20
reading_group = []

while True:
# Start ranging
    new_value = tof.ping()-50
    if new_value != 8141 and new_value != 8140:
        reading_group.append(new_value)
        if len(reading_group) > n:
            reading_group.pop(0)
        print(sum(reading_group)/ len(reading_group), " ", new_value)
    time.sleep(1)