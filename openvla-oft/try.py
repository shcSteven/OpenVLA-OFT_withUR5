import pickle
from experiments.robot.libero.run_libero_eval import GenerateConfig
from experiments.robot.openvla_utils import get_action_head, get_processor, get_proprio_projector, get_vla, get_vla_action
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM

# Instantiate config (see class GenerateConfig in experiments/robot/libero/run_libero_eval.py for definitions)
cfg = GenerateConfig(
    #pretrained_checkpoint = "moojink/openvla-7b-oft-finetuned-libero-spatial",
    pretrained_checkpoint = "/home/create/Documents/Haochen/openvla-7b+UR5+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--wrist_img--proprio_state--150000_chkpt",
    use_l1_regression = True,
    use_diffusion = False,
    use_film = False,
    num_images_in_input = 2,
    use_proprio = True,
    load_in_8bit = False,
    load_in_4bit = True,
    center_crop = True,
    num_open_loop_steps = NUM_ACTIONS_CHUNK,
    unnorm_key = "UR5",
)

# Load OpenVLA-OFT policy and inputs processor
vla = get_vla(cfg)
processor = get_processor(cfg)

# Load MLP action head to generate continuous actions (via L1 regression)
action_head = get_action_head(cfg, llm_dim=vla.llm_dim)

# Load proprio projector to map proprio to language embedding space
proprio_projector = get_proprio_projector(cfg, llm_dim=vla.llm_dim, proprio_dim=PROPRIO_DIM)

# Load sample observation:
#   observation (dict): {
#     "full_image": primary third-person image,
#     "wrist_image": wrist-mounted camera image,
#     "state": robot proprioceptive state,
#     "task_description": task description,
#   }
with open("experiments/robot/libero/sample_libero_spatial_observation.pkl", "rb") as file:
    observation = pickle.load(file)

print(observation.keys())
print("Task description:")
print(observation["task_description"])
print(observation['full_image'].shape)
print(observation['wrist_image'].shape)
print(observation['state'])

with open("/home/create/Documents/Haochen/demo_0_0.pkl", "rb") as file:
    observation = pickle.load(file)

print(observation.keys())
print("Task description:")
print(observation["task_description"])
print(observation['full_image'].shape)
print(observation['wrist_image'].shape)
print(observation['state'])

# Generate robot action chunk (sequence of future actions)
actions = get_vla_action(cfg, vla, processor, observation, observation["task_description"], action_head, proprio_projector)
print("Generated action chunk:")
for act in actions:
    print(act)
    