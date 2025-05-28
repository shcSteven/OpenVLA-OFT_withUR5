import rtde_control

import rtde_receive

import roslibpy

import numpy as np

import time

from scipy.spatial.transform import Rotation as R

import math

import cv2

from scipy.io import savemat

import os

import keyboard



# Connect to the Pi's rosbridge server over port 9090

client = roslibpy.Ros(host='192.168.1.2', port=9090) # The host is the Pi's IP address

client.run()

print(client.is_connected)

# Subscribe to the tendon state information topic

listener = roslibpy.Topic(client, '/spacenav/joy', 'sensor_msgs/msg/Joy')

def listener_callback(message):

    # print('Reading from /tendon_transmission_node/tendon_states')

    global axes 

    axes = message['axes']

    global buttons

    buttons = message['buttons']

    # Stop listening after the first message

    # listener.unsubscribe()

listener.subscribe(listener_callback)

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



#     return quaternion

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



gain_trans = .00005

gain_rot = .0001

# gain_trans = .005

# gain_rot = .01

speed = .5

acc = 2



arm_IP = "192.168.1.24"

# Setup robot with robot IP address

rtde_c = rtde_control.RTDEControlInterface(arm_IP)

rtde_r = rtde_receive.RTDEReceiveInterface(arm_IP)



config_list = rtde_r.getActualTCPPose()

config = np.array(config_list)



# vertical_position = config

print('Starting configuration: ',config)

time.sleep(0.5)



# Move robot along axes

# config[0] += 0.1

# config [1] -= 0.1

# config [2] -= 0.05

# print(config)

# rtde_c.moveL(config)

# config [2] -= 0.05

# rtde_c.moveL(config)



# Rotate robot along axes

# config[3] += 0.1

# config [4] -= 0.1

# config [5] += 0.1

# print(config)

# rtde_c.moveL(config)



# Move robot along along Z axis for 50 mm, non-blocking movement

# Default is False - blocking movement, True is non-blocking movement

# config [2] += 0.05



end_effector_trans = config[:3]

velocity = 0.02; acceleration = 0.02; dt_temp = 1.0/500; lookahead_time = 0.1; gain = 100



R_global = np.dot(np.dot(rotation_matrix_z(config[5]), 

                            rotation_matrix_y(config[4])), 

                            rotation_matrix_x(config[3]))



t_ref = time.time()





dataset_dir = "C:/Users/andre/Poli/Magistrale/Tesi/Python code/UR/user4_3"

goal='Feeding task with UR3'

data=[]



mat_file = "joystick_data.mat"

mat_data = {

    "end_effector_x": [],

    "end_effector_y": [],

    "end_effector_z": [],

    "rot_x": [],

    "rot_y": [],

    "rot_z": []

}

full_path = os.path.join(dataset_dir, mat_file)



if not os.path.exists(dataset_dir):

    os.makedirs(dataset_dir)



next_index = 1



loop_counter=0

action_tot=[0,0,0,0,0,0]





exit_flag = False

while not exit_flag :

    # Check if 'e' key is pressed to exit the loop 

    if keyboard.is_pressed('e'):

        exit_flag = True

    

    t0 = time.time()

    translation = np.dot(rotz(0),np.dot(rotx(180), [axis * gain_trans for axis in axes[:3]]))

    end_effector_trans = end_effector_trans + translation

    rotation = np.dot(roty(0),np.dot(rotx(180), [axis * gain_rot for axis in axes[3:]]))



    delta_theta_x = rotation[0]

    delta_theta_y = rotation[1]

    delta_theta_z = rotation[2]

    R_increment = np.dot(np.dot(rotation_matrix_x(delta_theta_x), 

                            rotation_matrix_y(delta_theta_y)), 

                            rotation_matrix_z(delta_theta_z))

    R_global = np.dot(R_increment, R_global)

    theta_x, theta_y, theta_z = euler_from_rotation_matrix(R_global)

    demand_rotvec = np.array([theta_x, theta_y, theta_z])

    

    config = np.concatenate((end_effector_trans, demand_rotvec)) 

    t_ref = time.time()

    print('\nconfig:',config)

    print(time.time() -t0)

    t1 = time.time()

    rtde_c.servoL(config, velocity, acceleration, dt_temp, lookahead_time, gain)

    print(time.time()-t1)





    mat_data["end_effector_x"].append(config[0])

    mat_data["end_effector_y"].append(config[1])

    mat_data["end_effector_z"].append(config[2])

    mat_data["rot_x"].append(config[3])

    mat_data["rot_y"].append(config[4])

    mat_data["rot_z"].append(config[5])



    

savemat(full_path, mat_data)

time.sleep(0.3)



import json

with open(os.path.join(dataset_dir, "dataset.json"), 'w') as f:

        json.dump(data, f, indent=4)

print('Dataset saved')

