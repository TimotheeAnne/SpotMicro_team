# SDA = pin.SDA_1
# SCL = pin.SCL_1
# SDA_1 = pin.SDA
# SCL_1 = pin.SCL

from adafruit_servokit import ServoKit
import board
import busio
import time
from approxeng.input.selectbinder import ControllerResource


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

print("Done initializing")



while(True):

    #### Set Initial Position ####

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


    #flhr.angle = 0
    #blhr.angle = 180

    #flhr.angle =int(input('Front left hip roll: '))
    #flhr.angle = int(input('Front left hip roll: '))
    
    #print("Current angle: " + str(flhp.angle))
    
    #brhp.angle=90
    
    #brkp.angle = int(input('Back left knee pitch: '))


    #flkp.angle = int(input('Back left knee pitch: '))

    #blhr.angle =int(input('Back left hip roll: '))



# while(True):
# #for degree in sweep:
# #    print("Degree: " + str(degree) + "\n")
#     flhp.angle=30
#     blhp.angle=30
#     print("Position at: " + str(30))
#     time.sleep(2)

#     flhp.angle=100
#     blhp.angle=100
#     print("Position at: " + str(100))
#     time.sleep(2)

#time.sleep(0.5)
#sweep = range(180,0, -1)
#for degree in sweep :
#    kit.servo[0].angle=degree
            
