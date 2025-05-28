from time import sleep
import sys
sys.path.append('../')

# Import from parent directory
from dynamixel_controller import Dynamixel

servo = Dynamixel(ID=10, descriptive_device_name="XM430 test motor", 
                    series_name="xm", baudrate=57600, port_name="/dev/ttyUSB0")

servo.begin_communication()

"""
servo.set_operating_mode("velocity")

for i in range(1):
    servo.write_velocity(100)
    sleep(1)

    servo.write_velocity(-100)
    sleep(1)
"""

servo.set_operating_mode("current")

initial_current = 30  # 100mA to close the gripper
current_threshold = 50  # 200mA when the gripper has a firm grip
max_current = 100  # Max current to try for gripping (500 mA)

# Gradually increase the current until the gripper holds the object
servo.write_current(initial_current)
print(f"Gripper is closing with {initial_current} mA...")

current = initial_current
while current < max_current:
    sleep(0.1)  # Wait a bit before checking the current
    actual_current = servo.read_current()  # Get the actual current feedback from the motor
    print(f"Actual Current: {actual_current} mA")

    # Check if the current exceeds a threshold (indicating resistance is being encountered)
    if actual_current >= current_threshold:  # 200 mA is an arbitrary threshold for gripping force
        print(f"Gripper has tightened. Stopping motor.")
        servo.write_current(actual_current)  # Stop the motor
        break  # Exit the loop

    # Increase the current incrementally to apply more force
    current += 5  # Increase current by 50 mA per loop
    servo.write_current(current)

# Wait for a few seconds before releasing the object
sleep(3)
servo.set_operating_mode("position")
servo.write_position(1400)  # Open the gripper
sleep(1)

servo.end_communication()
