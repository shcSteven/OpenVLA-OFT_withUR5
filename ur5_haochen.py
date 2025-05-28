import pyspacemouse # for 3Dconnexion
import time
import sys
if not './dynamixel_controller_main' in sys.path:
    sys.path.append('./dynamixel_controller_main')
# Import from parent directory
from dynamixel_controller import Dynamixel # for gripper control

import rtde_control
import rtde_receive
import numpy as np
from scipy.spatial.transform import Rotation as R
import math
import cv2
from scipy.io import savemat
import os
from pynput import keyboard
import threading
from multiprocessing import Process, Value, Manager
import json

demo_num = 888
images_3rd_path = f"/home/create/Documents/Haochen/data_collection/demo_{demo_num}/images_3rd"
images_wrist_path = f"/home/create/Documents/Haochen/data_collection/demo_{demo_num}/images_wrist"
UR5_records_path = f"/home/create/Documents/Haochen/data_collection/demo_{demo_num}/UR5_records"


# some constants
gain_trans = .0001
gain_rot = .0005
speed = .5
acc = 2
arm_IP = "192.168.1.20"
velocity = 0.02
acceleration = 0.02
dt_temp = 1.0/500
lookahead_time = 0.1
gain = 100

initial_current = 100  # gripper
current_threshold = 120
max_current = 200
initial_pwm = 550       # Starting PWM value (small positive)
max_pwm = 800           # Max PWM to prevent over-force (adjust based on your gripper)
current_threshold = 600  # mA, threshold to detect object gripping


def rotx(theta):
    theta = np.radians(theta)
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])

def roty(theta):
    theta = np.radians(theta)
    return np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

def rotz(theta):
    theta = np.radians(theta)
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])


def euler_to_quaternion(roll, pitch, yaw):
    
    # Calculate trigonometric functions once for efficiency
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    # Compute the quaternion components
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return w, x, y, z



def rotation_vector_to_euler(rotation_vector):

    theta = np.linalg.norm(rotation_vector)  # Magnitude of the rotation vector
    if theta == 0:
        return 0.0, 0.0, 0.0  # No rotation
    # Normalize the rotation vector
    axis = rotation_vector / theta
    # Compute the quaternion components from the rotation vector
    w = np.cos(theta / 2.0)
    x = axis[0] * np.sin(theta / 2.0)
    y = axis[1] * np.sin(theta / 2.0)
    z = axis[2] * np.sin(theta / 2.0)
    # Convert quaternion to Euler angles
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(2 * (w * y - z * x))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def euler_from_rotation_matrix(R):
    theta_y = np.arcsin(-R[2, 0])
    theta_x = np.arctan2(R[2, 1], R[2, 2])
    theta_z = np.arctan2(R[1, 0], R[0, 0])
    return theta_x, theta_y, theta_z


def rotation_matrix_x(theta):
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])



def rotation_matrix_y(theta):
    return np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])



def rotation_matrix_z(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])


"""
def gripper_grab(servo):
    servo.set_operating_mode("current")
    servo.write_current(initial_current)
    print(f"Gripper is closing with {initial_current} mA...")

    current = initial_current
    while current < max_current:
        time.sleep(0.1)  # Wait a bit before checking the current
        actual_current = servo.read_current()  # Get the actual current feedback from the motor
        print(f"Actual Current: {actual_current} mA")

        # Check if the current exceeds a threshold (indicating resistance is being encountered)
        if actual_current >= current_threshold:  # 200 mA is an arbitrary threshold for gripping force
            print(f"Gripper has tightened. Stopping motor.")
            #servo.write_current(actual_current)  # Stop the motor
            break  # Exit the loop

        # Increase the current incrementally to apply more force
        current += 10  # Increase current by 50 mA per loop
        servo.write_current(current)
"""
def gripper_grab(servo):
    #servo.torque_off()
    servo.set_operating_mode("pwm")  # Switch to PWM control mode
    #servo.torque_on()

    print(f"Gripper is closing with increasing PWM...")

    pwm_value = initial_pwm  # Start with low PWM value
    while pwm_value < max_pwm:
        time.sleep(0.05)  # Wait a bit before checking

        actual_current = servo.read_current()  # XL430 still allows reading actual current
        print(f"Actual Current: {actual_current} mA")

        # Check if gripping (i.e., resistance detected)
        if actual_current >= current_threshold:
            print(f"Gripper has tightened. Stopping motor.")
            servo.write_pwm(0)  # Stop applying force
            break

        # Increase the PWM gradually to apply more gripping force
        pwm_value += 10  # Increment PWM slowly
        if pwm_value > max_pwm:
            pwm_value = max_pwm

        servo.write_pwm(pwm_value)

    print("Gripper grabbing finished.")

def gripper_release(servo):
    print("release the gripper")
    servo.set_operating_mode("position")
    servo.write_position(2000)  # Open the gripper



def button_0(state, buttons, pressed_buttons):
    print("Button:", pressed_buttons)


def button_0_1(state, buttons, pressed_buttons):
    print("Buttons:", pressed_buttons)


def someButton(state, buttons):
    #print("Some button")
    pass


def record_episode(cap_3rd, cap_wrist, arm_IP, UR5_records, exit_flag, gripper_open):
    global images_3rd_path
    global images_wrist_path
    t0 = time.time()
    count_episodes = 0
    images_3rd_path_original = os.path.join(images_3rd_path, "original")
    images_3rd_path_resized = os.path.join(images_3rd_path, "resized")
    images_wrist_path_original = os.path.join(images_wrist_path, "original")
    images_wrist_path_resized = os.path.join(images_wrist_path, "resized")
    rtde_r = rtde_receive.RTDEReceiveInterface(arm_IP)
    while exit_flag.value == False:
        if time.time() - t0 >= count_episodes*0.2:
            print(time.time())
            for _ in range(5):  # Adjust number of iterations if needed
                _, frame_wrist = cap_wrist.read()
                _, frame_3rd = cap_3rd.read()
            #print("Capturing image...")
            # Save the captured image
            #print(frame_3rd.shape)
            cv2.imwrite(os.path.join(images_3rd_path_original, f"captured_image_3rd_{count_episodes}.jpg"), frame_3rd)
            cv2.imwrite(os.path.join(images_wrist_path_original, f"captured_image_wrist_{count_episodes}.jpg"), frame_wrist)
            cropped_image_3rd = frame_3rd[:, 140:640]  
            resized_image_3rd = cv2.resize(cropped_image_3rd, (256, 256), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(images_3rd_path_resized, f"captured_image_3rd_{count_episodes}.jpg"), resized_image_3rd)
            cropped_image_wrist = frame_wrist[:, 75:555]  
            resized_image_wrist = cv2.resize(cropped_image_wrist, (256, 256), interpolation=cv2.INTER_AREA)
            flipped_image_wrist = cv2.flip(resized_image_wrist, 0)
            cv2.imwrite(os.path.join(images_wrist_path_resized, f"captured_image_wrist_{count_episodes}.jpg"), flipped_image_wrist)

            joint_pos_list = rtde_r.getActualQ()
            pose_list = rtde_r.getActualTCPPose()
            joint_velo_list = rtde_r.getActualQd()
            pose_velo_list = rtde_r.getActualTCPSpeed()
            time_stamp = rtde_r.getTimestamp()
            #print("joint_pos:", joint_pos_list)
            #print("pose:", pose_list)
            #print("joint_velo:", joint_velo_list)
            UR5_records["joint_pos"].append(joint_pos_list)
            UR5_records["pose"].append(pose_list)
            UR5_records["joint_velo"].append(joint_velo_list)
            UR5_records["pose_velo"].append(pose_velo_list)
            UR5_records["time_stamp"].append(time_stamp)
            UR5_records["gripper_state"].append(gripper_open.value)

            count_episodes += 1
    del rtde_r


def callback(rtde_c, config, servo, exit_flag, gripper_open):
    button_arr = [pyspacemouse.ButtonCallback(0, button_0),
                  pyspacemouse.ButtonCallback([1], lambda state, buttons, pressed_buttons: print("Button: 1")),
                  pyspacemouse.ButtonCallback([0, 1], button_0_1), ]

    success = pyspacemouse.open(dof_callback=pyspacemouse.print_state, button_callback=someButton,
                                button_callback_arr=button_arr)
    if success:
        new_config = config
        z_axis_fixed = False
        while exit_flag.value == False:

            state = pyspacemouse.read()
            """
            R_global = np.dot(np.dot(rotation_matrix_z(new_config[5]), 
                                    rotation_matrix_y(new_config[4])), 
                                    rotation_matrix_x(new_config[3]))
            """
            end_effector_trans = new_config[:3]
        
            #axes = [state.x, state.y, state.z, state.roll, state.pitch, state.yaw]
            #axes = [state.x, state.y, state.z, state.pitch, state.yaw, state.roll]
            #axes = [state.x, state.y, state.z, state.roll, state.yaw, -state.pitch]
            if z_axis_fixed:
                axes = [state.x, state.y, 0.1*state.z, 0.0, state.yaw, 0.0]
            else:
                axes = [state.x, state.y, state.z, 0.0, state.yaw, 0.0]
            #print("axes:", axes)
            translation = np.dot(roty(0),np.dot(rotx(0), [axis * gain_trans for axis in axes[:3]]))
            #translation = [axis * gain_trans for axis in axes[:3]]
            #translation = [0,0,0]
            end_effector_trans = end_effector_trans + translation
            """
            rotation = np.dot(rotx(0),np.dot(roty(0), [-axis * gain_rot for axis in axes[3:]]))
            #rotation = [0,0,0]
            #rotation = [axis * gain_rot for axis in axes[3:]]

            delta_theta_x = rotation[0]
            delta_theta_y = rotation[1]
            delta_theta_z = rotation[2]
            
            #print("delta_theta:", delta_theta_x, delta_theta_y, delta_theta_z)

            R_increment = np.dot(np.dot(rotation_matrix_x(delta_theta_x), 
                                        rotation_matrix_y(delta_theta_y)), 
                                        rotation_matrix_z(delta_theta_z))
            print("R_increment:", R_increment)
            R_global = np.dot(R_increment, R_global)
            #R_global = np.dot(R_global, R_increment)
            
            theta_x, theta_y, theta_z = euler_from_rotation_matrix(R_global)
            demand_rotvec = np.array([theta_x, theta_y, theta_z])
            #demand_rotvec = np.array([-2.902167797, theta_y, 0.0])
            """
            R_fixed = R.from_rotvec(new_config[3:6])
            yaw_angle = state.yaw * gain_rot
            # Apply yaw around local Z (tool Z)
            R_yaw = R.from_rotvec([0, 0, yaw_angle])
            # Compose rotations: fixed downward + tool yaw
            R_global = R_fixed * R_yaw  # rotation in the correct frame

            # Get final rotation vector
            demand_rotvec = R_global.as_rotvec()

            new_config = np.concatenate((end_effector_trans, demand_rotvec)) 
            #print('\ncurrent config:', new_config)

            #print(time.time() -t0)

            try:
                rtde_c.servoL(new_config, velocity, acceleration, dt_temp, lookahead_time, gain)
            except Exception as e:
                print(f"Error in servoL command: {e}")
            """
            if state.buttons[0] == 1:
                gripper_open.value = False
                gripper_grab(servo)
                z_axis_fixed = True
                time.sleep(0.01)
            elif state.buttons[1] == 1:
                gripper_open.value = True
                gripper_release(servo)
                time.sleep(0.01)
            """
            if state.buttons[1] == 1 and gripper_open.value == True:
                gripper_open.value = False
                gripper_grab(servo)
                time.sleep(0.01)
            elif state.buttons[1] == 1 and gripper_open.value == False:
                gripper_open.value = True
                gripper_release(servo)
                time.sleep(0.01)
            if state.buttons[0] == 1:
                z_axis_fixed = True
            

            #print(time.time()-t1)
            time.sleep(0.001)

            #if state.buttons[1] == 1:
                #exit_flag = True
            




if __name__ == "__main__":
    exit_flag = Value('b', False)  # Shared boolean flag
    gripper_open = Value('b', True) # Shared boolean flag, True when gripper is open and not grabbing anything
    manager = Manager()
    UR5_records = manager.dict({
        "joint_pos": manager.list(),
        "pose": manager.list(),
        "joint_velo": manager.list(),
        "pose_velo": manager.list(),
        "time_stamp": manager.list(),
        "gripper_state": manager.list()
    })
    

    os.makedirs(os.path.join(images_3rd_path, "original"), exist_ok=True)
    os.makedirs(os.path.join(images_3rd_path, "resized"), exist_ok=True)
    os.makedirs(os.path.join(images_wrist_path, "original"), exist_ok=True)
    os.makedirs(os.path.join(images_wrist_path, "resized"), exist_ok=True)
    os.makedirs(UR5_records_path, exist_ok=True)
    #listener = keyboard.Listener(on_press=on_press_exit)
    #listener.start()  # This runs the listener in a separate thread
    cap_wrist = cv2.VideoCapture(2)
    if not cap_wrist.isOpened():
        print("Camera index 2 failed, trying index 4...")
        cap_wrist = cv2.VideoCapture(3)
    cap_3rd = cv2.VideoCapture(0)
    if not cap_3rd.isOpened():
        print("Camera index 0 failed, trying index 1...")
        cap_3rd = cv2.VideoCapture(1)

    # Check if the camera opened successfully
    if not cap_3rd.isOpened():
        print("Error: Could not open 3rd view webcam.")
        exit()
    if not cap_wrist.isOpened():
        print("Error: Could not open wrist view webcam.")
        exit()
    
    servo = Dynamixel(ID=1, descriptive_device_name="XM430 test motor", 
                    series_name="xm", baudrate=57600, port_name="/dev/ttyUSB0")

    servo.begin_communication()
    servo.set_operating_mode("position")
    servo.write_position(2000)  # Open the gripper

    # Setup robot with robot IP address
    rtde_c = rtde_control.RTDEControlInterface(arm_IP)
    #rtde_r = rtde_receive.RTDEReceiveInterface(arm_IP)

    #config_list = rtde_r.getActualTCPPose()
    #print("initial config_list:", config_list)
    #config = np.array(config_list)
    
    try:
        #rtde_c.moveL([-0.35875322, -0.21916963, 0.48547606, -2.82960053, 1.30837597,-0.00579543], speed, acceleration)
        rtde_c.setPayload(1.2, [0, 0, 0.05])
        rtde_c.moveL([-0.35875322, -0.21916963, 0.48547606, -2.902167797, 1.20196913, 0.0], speed, acceleration)
        config = np.array([-0.35875322, -0.21916963, 0.48547606, -2.902167797, 1.20196913, 0.0])
        print('Starting configuration: ',config)
    except Exception as e:
        print(f"Error in servoL command: {e}")
    
    time.sleep(1)

    p1 = Process(target=callback, args=(rtde_c, config, servo, exit_flag, gripper_open))
    p2 = Process(target=record_episode, args=(cap_3rd, cap_wrist, arm_IP ,UR5_records, exit_flag, gripper_open))
    p1.start()
    p2.start()
    
    # Let the threads run for 30 seconds
    time.sleep(35)

    # Stop threads by setting the flag to True
    exit_flag.value = True

    # Wait for threads to finish
    p1.join()
    p2.join()

    # exits
    servo.end_communication()
    cap_3rd.release()
    cap_wrist.release()
    cv2.destroyAllWindows()

    if 'rtde_c' in locals():
        print("Stopping RTDE control")
        rtde_c.stopScript()    
    #if 'rtde_r' in locals():
        #print("Closing RTDE receive")
        #del rtde_r  # This should properly release the receive interface    print(“Cleanup done.“)

    #UR5_records = dict(UR5_records)
    # print(UR5_records)

    UR5_records_normal = {
        "joint_pos": list(UR5_records["joint_pos"]),
        "pose": list(UR5_records["pose"]),
        "joint_velo": list(UR5_records["joint_velo"]),
        "pose_velo": list(UR5_records["pose_velo"]),
        "time_stamp": list(UR5_records["time_stamp"]),
        "gripper_state": list(UR5_records["gripper_state"])
    }
    with open(os.path.join(UR5_records_path, "data.json"), "w") as json_file:
        json.dump(UR5_records_normal, json_file, indent=4)  # `indent=4` makes it readable

    
