from time import sleep
import sys
sys.path.append('../')

# Import from parent directory
from dynamixel_controller import Dynamixel

servo = Dynamixel(ID=[1,2], descriptive_device_name="XM430 test motor", 
                    series_name=["xm", "xm"], baudrate=1000000, port_name="/dev/ttyUSB0")


servo.begin_communication()

servo.set_operating_mode("position", ID = "all")

servo.write_position(0, ID="all")
sleep(1)

servo.write_position(2000, ID=[1,2])
sleep(1)

servo.write_position(0, ID=1)
sleep(1)

servo.set_operating_mode("velocity", ID = 1)

servo.write_velocity(100, ID = 1)
sleep(1)

servo.write_velocity(-100, ID = 1)
sleep(1)

servo.end_communication()
