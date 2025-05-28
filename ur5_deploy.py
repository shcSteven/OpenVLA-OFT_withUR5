import logging
import os
import socket
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import tqdm

# Append current directory so that interpreter can find experiments.robot
sys.path.append("./openvla-oft/experiments/robot")
if not '/home/create/Documents/Haochen/dynamixel_controller_main' in sys.path:
    sys.path.append('/home/create/Documents/Haochen/dynamixel_controller_main')
# Import from parent directory
from dynamixel_controller import Dynamixel # for gripper control
import rtde_control
import rtde_receive
import pickle
import cv2
import numpy as np
import imageio
from scipy.spatial.transform import Rotation as R

from experiments.robot.openvla_utils import (
    get_action_from_server,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_image_resize_size,
    set_seed_everywhere,
)


IDX = 10


speed = 2
arm_IP = "192.168.1.20"
acceleration = 0.03

initial_current = 100  # gripper
current_threshold = 120
max_current = 200
initial_pwm = 400       # Starting PWM value (small positive)
max_pwm = 600           # Max PWM to prevent over-force (adjust based on your gripper)
current_threshold = 440  # mA, threshold to detect object gripping

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)



@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                    # Number of actions to execute open-loop before requerying policy
    max_steps: int = 200                          # Max number of steps to execute in episode

    use_vla_server: bool = True                      # Whether to query remote VLA server for actions
    vla_server_url: Union[str, Path] = "20.218.232.171"            # Remote VLA server URL (set to 127.0.0.1 if on same machine)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./UR5_deploy/logs"        # Local directory for eval logs

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on


def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.use_vla_server, (
        "Must use VLA server (server-client interface) to query model and get actions! Please set --use_vla_server=True"
    )


def setup_logging(cfg: GenerateConfig):
    """Set up logging to file."""
    # Create run ID
    run_id = f"EVAL-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    # Set up local logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    print(message)
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def save_rollout_video(rollout_dir ,rollout_images, idx, success, task_description, log_file=None, notes=None):
    """Saves an MP4 replay of an episode."""
    #rollout_dir = f"./UR5_deploy/videos"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    filetag = f"{rollout_dir}/episode={idx}--success={success}--task={processed_task_description}"
    if notes is not None:
        filetag += f"--{notes}"
    mp4_path = f"{filetag}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=12)
    
    for img in rollout_images:
        video_writer.append_data(img)

    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


def get_server_endpoint(cfg: GenerateConfig):
    """Get the server endpoint for remote inference."""
    ip_address = socket.gethostbyname(cfg.vla_server_url)
    return f"http://{ip_address}:8777/act"

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
        time.sleep(0.1)  # Wait a bit before checking

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


def run_episode_ur5(
    cfg: GenerateConfig,
    rtde_c,
    servo,
    cap_3rd,
    cap_wrist,
    task_description: str,
    server_endpoint: str,
    log_file=None,
):
    # Define control frequency
    STEP_DURATION_IN_SEC = 1.0 / 25.0
    # Reset environment
    servo.set_operating_mode("position")
    servo.write_position(2000)  # Open the gripper
    try:
        rtde_c.setPayload(1.2, [0, 0, 0.05])
        rtde_c.moveL([-0.35875322, -0.21916963, 0.48547606, -2.902167797, 1.20196913, 0.0], speed, acceleration)
        cur_state = np.array([-0.35875322, -0.21916963, 0.48547606, -2.902167797, 1.20196913, 0.0, 0, 1])
        #print('Starting configuration: ',config)
    except Exception as e:
        print(f"Error in servoL command: {e}")
    cur_gripper_state = 1
    time.sleep(2)

    replay_images_3rd = []
    replay_images_wrist = []

    # Initialize action queue
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    episode_start_time = time.time()
    total_model_query_time = 0.0
    t = 0

    try:
        while t < cfg.max_steps:
            step_start_time = time.time()

            for _ in range(5):  # Adjust number of iterations if needed
                _, frame_wrist = cap_wrist.read()
                _, frame_3rd = cap_3rd.read()
            cropped_image_3rd = frame_3rd[:, 140:640] 
            resized_image_3rd = cv2.resize(cropped_image_3rd, (256, 256), interpolation=cv2.INTER_AREA) 
            cropped_image_wrist = frame_wrist[:, 75:555] 
            resized_image_wrist = cv2.resize(cropped_image_wrist, (256, 256), interpolation=cv2.INTER_AREA)
            flipped_image_wrist = cv2.flip(resized_image_wrist, 0)
            resized_image_3rd = cv2.cvtColor(resized_image_3rd, cv2.COLOR_BGR2RGB)
            flipped_image_wrist = cv2.cvtColor(flipped_image_wrist, cv2.COLOR_BGR2RGB)
            # Save images for replay
            replay_images_3rd.append(resized_image_3rd)
            replay_images_wrist.append(flipped_image_wrist)

            if len(action_queue) == 0:
                # Prepare observation
                observation = {
                    "full_image": resized_image_3rd,
                    "wrist_image": flipped_image_wrist,
                    "state": cur_state,
                    "instruction": task_description,
                }

                # Query model to get action
                log_message("Requerying model...", log_file)
                model_query_start_time = time.time()
                actions = get_action_from_server(observation, server_endpoint)
                #print(actions)
                actions = actions[: cfg.num_open_loop_steps]
                total_model_query_time += time.time() - model_query_start_time
                action_queue.extend(actions)

            # Get action from queue
            action = action_queue.popleft()
            #print(f"Action: {action}")
            log_message("-----------------------------------------------------", log_file)
            log_message(f"t: {t}", log_file)
            log_message(f"action: {action}", log_file)

            # Execute action in environment
            rx, ry, rz = cur_state[3:6]
            rot = R.from_rotvec([rx, ry, rz])
            roll, pitch, yaw = rot.as_euler('xyz', degrees=False)
            new_roll = roll + action[3]
            new_pitch = pitch + action[4]
            new_yaw = yaw + action[5]
            new_x = cur_state[0] + action[0]
            new_y = cur_state[1] + action[1]
            new_z = cur_state[2] + action[2]
            new_rx, new_ry, new_rz = R.from_euler('xyz', [new_roll, new_pitch, new_yaw]).as_rotvec()
            cur_state = np.array([new_x, new_y, new_z, new_rx, new_ry, new_rz])

            #cur_state = cur_state[:6] + action[:6]
            gripper_state = round(action[6])
            rtde_c.moveL(cur_state, speed, acceleration)
            cur_state = np.append(cur_state, 0)
            cur_state = np.append(cur_state, gripper_state)
            if gripper_state != cur_gripper_state:
                if gripper_state == 0:
                    gripper_grab(servo)
                else:
                    gripper_release(servo)
                    for _ in range(15):
                        for _ in range(5):  # Adjust number of iterations if needed
                            _, frame_wrist = cap_wrist.read()
                            _, frame_3rd = cap_3rd.read()
                        cropped_image_3rd = frame_3rd[:, 140:640] 
                        resized_image_3rd = cv2.resize(cropped_image_3rd, (256, 256), interpolation=cv2.INTER_AREA) 
                        cropped_image_wrist = frame_wrist[:, 75:555] 
                        resized_image_wrist = cv2.resize(cropped_image_wrist, (256, 256), interpolation=cv2.INTER_AREA)
                        flipped_image_wrist = cv2.flip(resized_image_wrist, 0)
                        resized_image_3rd = cv2.cvtColor(resized_image_3rd, cv2.COLOR_BGR2RGB)
                        flipped_image_wrist = cv2.cvtColor(flipped_image_wrist, cv2.COLOR_BGR2RGB)
                        replay_images_3rd.append(resized_image_3rd)
                        replay_images_wrist.append(flipped_image_wrist)
                        time.sleep(0.1)
                    break
                cur_gripper_state = gripper_state
            t += 1
            #print(t)
            
            # Sleep until next timestep
            step_elapsed_time = time.time() - step_start_time
            if step_elapsed_time < STEP_DURATION_IN_SEC:
                time_to_sleep = STEP_DURATION_IN_SEC - step_elapsed_time
                log_message(f"Sleeping {time_to_sleep} sec...", log_file)
                time.sleep(time_to_sleep)


    except (KeyboardInterrupt, Exception) as e:
        if isinstance(e, KeyboardInterrupt):
            log_message("\nCaught KeyboardInterrupt: Terminating episode early.", log_file)
        else:
            log_message(f"\nCaught exception: {e}", log_file)     

    episode_end_time = time.time()

    # Get success feedback from user
    user_input = input("Success? Enter 'y' or 'n': ")
    success = True if user_input.lower() == "y" else False   
    # Calculate episode statistics
    episode_stats = {
        "success": success,
        "total_steps": t,
        "model_query_time": total_model_query_time,
        "episode_duration": episode_end_time - episode_start_time,
    }
    return episode_stats, replay_images_3rd, replay_images_wrist
    

@draccus.wrap()
def eval_ur5(cfg: GenerateConfig) -> None:

    cap_wrist = cv2.VideoCapture(2)
    if not cap_wrist.isOpened():
        print("Camera index 2 failed, trying index 3...")
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
    rtde_c = rtde_control.RTDEControlInterface(arm_IP)

    # Validate configuration
    validate_config(cfg)

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(cfg)

    # Get expected image dimensions
    #resize_size = get_image_resize_size(cfg)
    #print(f"Expected image dimensions: {resize_size}")

    # Get server endpoint for remote inference
    server_endpoint = get_server_endpoint(cfg)
    # Initialize task description
    task_description = "wipe the spilled coke with the sponge"
    episode_stats, replay_images_3rd, replay_images_wrist = run_episode_ur5(cfg, rtde_c, servo, cap_3rd, cap_wrist, task_description, server_endpoint, log_file=log_file)
    print("Episode statistics:")
    print(f"Success: {episode_stats['success']}")
    print(f"Total steps: {episode_stats['total_steps']}")
    print(f"Model query time: {episode_stats['model_query_time']:.2f} sec")
    print(f"Episode duration: {episode_stats['episode_duration']:.2f} sec")
    log_message(f"Success: {episode_stats['success']}", log_file)
    log_message(f"Total steps: {episode_stats['total_steps']}", log_file)
    log_message(f"Model query time: {episode_stats['model_query_time']:.2f} sec", log_file)
    log_message(f"Episode duration: {episode_stats['episode_duration']:.2f} sec", log_file)

    # Save video of the episode
    success = episode_stats["success"]
    mp4_path_3rd = save_rollout_video(f"./UR5_deploy/videos/3rd",replay_images_3rd, IDX, success, task_description, log_file=log_file)
    mp4_path_wrist = save_rollout_video(f"./UR5_deploy/videos/wrist", replay_images_wrist, IDX, success, task_description, log_file=log_file)
    
    """
    with open("/home/create/Documents/Haochen/demo_0_0.pkl", "rb") as file:
        observation = pickle.load(file)

    print(observation.keys())
    print("Task description:")
    print(observation["task_description"])
    print(observation['full_image'].shape)
    print(observation['wrist_image'].shape)
    print(observation['state'])

    observation["instruction"] = task_description

    actions = get_action_from_server(observation, server_endpoint)
    print("Generated action chunk:")
    for act in actions:
        print(act)
    """
    #log_message(f"Total model query time: {episode_stats['model_query_time']:.2f} sec", log_file)
    #log_message(f"Total episode elapsed time: {episode_stats['episode_duration']:.2f} sec", log_file)


    # Close log file
    if log_file:
        log_file.close()
    
    servo.end_communication()
    cap_3rd.release()
    cap_wrist.release()
    cv2.destroyAllWindows()

    if 'rtde_c' in locals():
        print("Stopping RTDE control")
        rtde_c.stopScript()   


if __name__ == "__main__":
    eval_ur5()