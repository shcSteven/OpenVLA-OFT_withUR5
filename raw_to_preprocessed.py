import os
import random
import json
from scipy.spatial.transform import Rotation as R
import numpy as np
import cv2

def get_demo_folders(directory, num_folders=5):
    # List all folders in the directory
    folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    
    # Filter only folders named in the format "demo_{i}"
    demo_folders = [f for f in folders if f.startswith("demo_") and f[5:].isdigit()]
    
    # Extract indices
    indices = sorted(int(f[5:]) for f in demo_folders)

    # Get 5 random indices (if there are at least 5 folders)
    # 
    #random_indices = random.sample(indices, min(num_folders, len(indices)))
    random_indices = [100,101,102,103,104,105,106,107,108,109]  # For testing purposes, we use fixed indices
    indices = list(set(indices) - set(random_indices))

    return sorted(indices), sorted(random_indices)


def last_index_of_zero(lst):
    try:
        return len(lst) - 1 - lst[::-1].index(0)
    except ValueError:
        return -1  # Return -1 if there is no 0 in the list


def first_1_after_first_0_sequence(seq):
    found_zero_block = False
    for i in range(1, len(seq)):
        if not found_zero_block and seq[i - 1] == 1 and seq[i] == 0:
            found_zero_block = True
        elif found_zero_block and seq[i] == 1:
            return i
    return -1  # if not found


def compute_delta_pose(pose_list):
    delta_list = []

    for i in range(1, len(pose_list)):
        prev_pose = np.array(pose_list[i-1])  # Convert previous pose to array
        curr_pose = np.array(pose_list[i])   # Convert current pose to array

        # Compute translation delta (difference in position)
        delta_translation = curr_pose[:3] - prev_pose[:3]
        
        prev_roll_pitch_yaw = R.from_rotvec(prev_pose[3:6]).as_euler('xyz', degrees=False)
        curr_roll_pitch_yaw = R.from_rotvec(curr_pose[3:6]).as_euler('xyz', degrees=False)

        delta_rotation = curr_roll_pitch_yaw - prev_roll_pitch_yaw
        
        # Combine translation and rotation deltas
        delta = np.concatenate([delta_translation, delta_rotation])
        
        delta_list.append(delta)

    # The last delta is [0, 0, 0, 0, 0, 0] as there is no next pose
    delta_list.append([0, 0, 0, 0, 0, 0])

    return delta_list


def concatenate_with_padding(np_array, my_list, padding_value=0):
    # Ensure that the length of the list matches the number of rows in the numpy array
    assert len(np_array) == len(my_list), "Length of the list must match the number of rows in the np_array"

    # Create a padding array with the same number of rows as the np_array and 1 column
    padding_column = np.full((np_array.shape[0], 1), padding_value)
    
    # Convert the list to a numpy array with shape (N, 1)
    list_column = np.array(my_list).reshape(-1, 1)
    
    # Concatenate the original array, padding column, and the list column
    result = np.hstack((np_array, padding_column, list_column))
    
    return result


def convert_to_rpy(pose_list):
    # Initialize a list to hold the converted poses
    converted_poses = []

    for pose in pose_list:
        # Extract translation (x, y, z) and rotation vector (rx, ry, rz)
        x, y, z, rx, ry, rz = pose

        # Convert the rotation vector (rx, ry, rz) to roll-pitch-yaw using Rotation from scipy
        rotation = R.from_rotvec([rx, ry, rz])  # Create a rotation object from the rotation vector
        rpy = rotation.as_euler('xyz', degrees=False)  # Convert to roll, pitch, yaw in radians

        # Combine the translation and RPY values
        converted_pose = np.array([x, y, z, *rpy])  # Concatenate translation and RPY
        converted_poses.append(converted_pose)

    # Convert the list of converted poses to a numpy array
    return np.array(converted_poses, dtype=np.float32)


def preprocess(indices, set_name='train'):
    for i in indices:
    # read json
        with open(os.path.join(raw_data_path, f'demo_{i}/UR5_records/data.json'), "r", encoding="utf-8") as file:
            data = json.load(file)  # Load JSON as a dictionary
            #joint_pos = data['joint_pos']
            pose = data['pose']
            #joint_velo = data['joint_velo']
            #pose_velo = data['pose_velo']
            #time_stamp = data['time_stamp']
            gripper_state = data['gripper_state']

            index = last_index_of_zero(gripper_state)
            pose = pose[:index+2]
            gripper_state = gripper_state[:index+2]
            #index = first_1_after_first_0_sequence(gripper_state)
            #pose = pose[:index+1]
            #gripper_state = gripper_state[:index+1]

            delta_pose = np.array(compute_delta_pose(pose), dtype=np.float32)

            indices = np.where(np.all(np.abs(delta_pose) < 1e-3, axis=1))[0]
            indices_big = np.where(np.all(np.abs(delta_pose) > 1, axis=1))[0]
            
            filtered_indices = [idx for i, idx in enumerate(indices) if gripper_state[idx] == gripper_state[idx - 1]]


            full_list = set(range(index + 2)) 
            #full_list = set(range(index + 1)) 
            needed_indices = sorted(list((full_list - set(filtered_indices))))


            needed_pose = [data['pose'][i] for i in needed_indices]
            needed_time_stamp = [data['time_stamp'][i] for i in needed_indices]
            needed_gripper_state = [data['gripper_state'][i] for i in needed_indices]

            
            converted_needed_pose = convert_to_rpy(needed_pose)
            state = concatenate_with_padding(converted_needed_pose, needed_gripper_state)

            needed_delta_pose = np.array(compute_delta_pose(needed_pose), dtype=np.float32)

            action_gripper_state = needed_gripper_state[1:] + [needed_gripper_state[-1]]
            action_gripper_state = np.array(action_gripper_state).reshape(-1, 1) 
            action = np.hstack((needed_delta_pose, action_gripper_state))

        print("in demo", i)
        print("indices_big", indices_big)
        print("filtered_indices", filtered_indices)
        print("needed_indices", needed_indices)

        episode = []
        for j,idx in enumerate(needed_indices):
            image_3rd = cv2.imread(os.path.join(raw_data_path, f'demo_{i}/images_3rd/resized/captured_image_3rd_{idx}.jpg'))
            image_3rd_rgb = cv2.cvtColor(image_3rd, cv2.COLOR_BGR2RGB)
            image_wrist = cv2.imread(os.path.join(raw_data_path, f'demo_{i}/images_wrist/resized/captured_image_wrist_{idx}.jpg'))
            image_wrist_rgb = cv2.cvtColor(image_wrist, cv2.COLOR_BGR2RGB)
            
            if (0 <= i < 50) or (100 <= i < 105):
                episode.append({
                    'image': image_3rd_rgb.astype(np.uint8),
                    'wrist_image': image_wrist_rgb.astype(np.uint8),
                    'state': state[j].astype(np.float32),
                    'action': action[j].astype(np.float32),
                    'language_instruction': 'put orange in the plate',
                })
            else:
                episode.append({
                    'image': image_3rd_rgb.astype(np.uint8),
                    'wrist_image': image_wrist_rgb.astype(np.uint8),
                    'state': state[j].astype(np.float32),
                    'action': action[j].astype(np.float32),
                    'language_instruction': 'put milk in the plate',
                })
            """
            episode.append({
                    'image': image_3rd_rgb.astype(np.uint8),
                    'wrist_image': image_wrist_rgb.astype(np.uint8),
                    'state': state[j].astype(np.float32),
                    'action': action[j].astype(np.float32),
                    'language_instruction': 'wipe the spilled coke with the sponge',
                })
            """
        save_path = os.path.join(f'./data/{set_name}', f'demo_{i}.npy')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, episode)

raw_data_path = "./data_collection_orange_milk"
train_indices, val_indices = get_demo_folders(raw_data_path,10)

print("val_indices", val_indices)
preprocess(train_indices, 'train')
preprocess(val_indices, 'val')

        
