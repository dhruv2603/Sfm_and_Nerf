import os
import cv2
import PIL
import matplotlib.pyplot as plt
import shutil
from natsort import natsorted
import json
import numpy as np

# Define the paths
source_folder = './images'
test_folder = './test'
train_folder = './train'
val_folder = './val'

# Create the target folders if they don't exist
os.makedirs(test_folder, exist_ok=True)
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

# Get the list of all image filenames
image_files = sorted(os.listdir(source_folder))

# Total number of images
idx_test = []
idx_train = []
idx_val = []

for i in range(151):
    if((i+7)%10 == 0):
        idx_val.append(i)
    elif(i%2==0 and i!= 150):
        idx_test.append(i)
    else:
        idx_train.append(i)
print("Test:")
print(idx_test)
print("Val:")
print(idx_val)
print("Train:")
print(idx_train)

# # Function to move the images to the corresponding folders
# def move_images(files, destination_folder):
#     for file in files:
#         src = os.path.join(source_folder, file)
#         dst = os.path.join(destination_folder, file)
#         shutil.move(src, dst)


# test_files = [image_files[i] for i in idx_test]
# move_images(test_files, test_folder)

# # Select images for the train set (next 60 images)
# train_files = [image_files[i] for i in idx_train]
# move_images(train_files, train_folder)

# # Select images for the validation set (remaining 15 images)
# val_files = [image_files[i] for i in idx_val]
# move_images(val_files, val_folder)

with open("transforms.json",'r') as f:
        data = json.load(f)
        camera_angle_x = data.get("camera_angle_x",0)
transform_test = {"camera_angle_x":camera_angle_x,"frames":[]}
transform_train = {"camera_angle_x":camera_angle_x,"frames":[]}
transform_val = {"camera_angle_x":camera_angle_x,"frames":[]}

for i,frame in enumerate(data["frames"]):
    if i in idx_test:
        transform_test["frames"].append({"file_path":frame["file_path"],"transform_matrix":frame["transform_matrix"]})
    elif i in idx_train:
        transform_train["frames"].append({"file_path":frame["file_path"],"transform_matrix":frame["transform_matrix"]})
    elif i in idx_val:
        transform_val["frames"].append({"file_path":frame["file_path"],"transform_matrix":frame["transform_matrix"]})

print(transform_test)
with open("transforms_test.json", 'w') as f:
    json.dump(transform_test, f, indent=4)

with open("transforms_train.json", 'w') as f:
    json.dump(transform_train, f, indent=4)

with open("transforms_val.json", 'w') as f:
    json.dump(transform_val, f, indent=4)