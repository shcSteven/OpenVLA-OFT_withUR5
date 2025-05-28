import cv2
import numpy as np
import os
import pickle as pkl

demo_0 = np.load("/home/create/Documents/Haochen/data/train/demo_0.npy", allow_pickle=True)
print(demo_0[0].keys())
image = demo_0[0]['image']
wrist_image = demo_0[0]['wrist_image']
state = demo_0[0]['state']
language_instruction = demo_0[0]['language_instruction']

demo_0_0 = {}
demo_0_0['full_image'] = image
demo_0_0['wrist_image'] = wrist_image
demo_0_0['state'] = state
demo_0_0['task_description'] = language_instruction

#with open('demo_0_0.pkl', 'wb') as f:
    #pkl.dump(demo_0_0, f)

print(demo_0[0]['action'])
print(demo_0[1]['action'])
print(demo_0[2]['action'])
print(demo_0[3]['action'])
print(demo_0[4]['action'])
print(demo_0[5]['action'])
print(demo_0[6]['action'])
print(demo_0[7]['action'])