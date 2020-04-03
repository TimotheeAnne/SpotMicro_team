# SDA = pin.SDA_1
# SCL = pin.SCL_1
# SDA_1 = pin.SDA
# SCL_1 = pin.SCL

from adafruit_servokit import ServoKit
import board
import busio
import time
from approxeng.input.selectbinder import ControllerResource
import numpy as np


# On the Jetson Nano
# Bus 0 (pins 28,27) is board SCL_1, SDA_1 in the jetson board definition file
# Bus 1 (pins 5, 3) is board SCL, SDA in the jetson definition file
# Default is to Bus 1; We are using Bus 0, so we need to construct the busio first ...


print("Initializing Servos")
i2c_bus0=(busio.I2C(board.SCL_1, board.SDA_1))
print("Initializing ServoKit")
kit = ServoKit(channels=16, i2c=i2c_bus0)
# kit[0] is the bottom servo
# kit[1] is the top servo

#### Initialize Motors ####

# Front left leg initialize
flhr = kit.servo[13]
flhp = kit.servo[14]
flkp = kit.servo[15]

# Back left leg initialize
blhr = kit.servo[9]
blhp = kit.servo[10]
blkp = kit.servo[11]

# Front right leg initialize
frhr = kit.servo[1]
frhp = kit.servo[2]
frkp = kit.servo[3]

# Back right leg initialize
brhr = kit.servo[5]
brhp = kit.servo[6]
brkp = kit.servo[7]

#### Set Initial Position ####


print("Done initializing")

def straighten():
    # Straighten legs 

    # Front left leg zero degrees
    flhr.angle = 133
    flhp.angle = 70
    flkp.angle = 100

    # Back left leg zero degrees
    blhr.angle = 95
    blhp.angle = 70
    blkp.angle = 142

    # Front right leg zero degrees
    frhr.angle = 90
    frhp.angle = 83
    frkp.angle = 85

    # Back right leg zero degrees
    brhr.angle = 95
    brhp.angle = 125
    brkp.angle = 85



straighten()

time.sleep(4)

# flhr.angle = 133
# flhp.angle = 120
# flkp.angle = 0



def setPos(angle, flip = False):
    
    if flip == True:
        angle = - angle
    
    if angle < 0:
        angle = 0
    elif angle > 180:
        angle = 180

    return angle 

sweepHip = np.linspace(0, 70, 20)
sweepKnee = np.linspace(0, 100, 20)

startPosHip = flhp.angle

for degree in sweepHip:
    flhp.angle = setPos(startPosHip + degree)
    time.sleep(0.05)

# # Front left squat
# flhp.angle = setPos(flhp.angle + 70)
# flkp.angle = setPos(flkp.angle - 100)

# # Back left squat
# blhp.angle = setPos(blhp.angle + 70)
# blkp.angle = setPos(blkp.angle - 100)

# # Front right squat 
# frhp.angle = setPos(frhp.angle - 70)
# frkp.angle = setPos(frkp.angle + 100)

# # Back right squat 
# brhp.angle = setPos(brhp.angle - 70)
# brkp.angle = setPos(brkp.angle + 100)



# while(True):

#     #### Set Initial Position ####

#     # Front left leg zero degrees
#     flhr.angle = 133
#     flhp.angle = 70
#     flkp.angle = 100

#     # Back left leg zero degrees
#     blhr.angle = 95
#     blhp.angle = 70
#     blkp.angle = 142

#     # Front right leg zero degrees
#     frhr.angle = 90
#     frhp.angle = 83
#     frkp.angle = 85

#     # Back right leg zero degrees
#     brhr.angle = 95
#     brhp.angle = 125
#     brkp.angle = 85

