import imageio
import os
import json
from natsort import natsorted
import imageio.v3 as imageio
import cv2
import numpy as np


def loadDataset(data_path, mode):
    """
    Input:
        data_path: dataset path
        mode: train or test
    Outputs:
        camera_info: image width, height, camera matrix
        images: images
        pose: corresponding camera pose in world frame
    """
    # Specify the image path
    img_path = os.path.join(data_path, mode)

    # Load the json file
    with open(os.path.join(data_path, "transforms_" + mode + ".json"), "r") as f:
        data = json.load(f)

    # Extract the camera angle
    camera_angle_x = data.get("camera_angle_x", 0)

    # Load the images
    files = natsorted(os.listdir(img_path))

    # Resize the images
    images = [
        cv2.resize(
            imageio.imread(os.path.join(img_path, i)),
            (400, 400),
            interpolation=cv2.INTER_LINEAR,
        )
        for i in files
        if i.endswith(".png") and "_depth_" not in i
    ]
    images = (np.array(images) / 255.0).astype(np.float32)

    # Make 3 channel instead of 4 channel
    if images.shape[-1] == 4:  # RGBA --> RGB
        images = images[..., :3] * images[..., -1:] + (1 - images[..., -1:])

    # Specify the image size
    width = images[0].shape[1]
    height = images[0].shape[0]

    # Camera matrix for a pin-hole camera model
    f_x = 0.5 * width / np.tan(camera_angle_x * 0.5)
    f_y = f_x
    c_x = width / 2
    c_y = height / 2
    camera_matrix = np.array([[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]])
    print(camera_matrix)

    # Extract the camera poses
    theta = -np.deg2rad(90)
    R_x = np.array([[1, 0, 0], [0.0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
    pose = []

    for frame in data["frames"]:
        homogeneous = np.array(frame["transform_matrix"])
        #rotation = homogeneous[0:3, 0:3]
        #rotation_new = rotation @ R_x
        #homogeneous[0:3, 0:3] = rotation_new
        transform_matrix = homogeneous
        pose.append(transform_matrix)

    camera_info = {"width": width, "height": height, "camera_matrix": camera_matrix}

    return images, np.array(pose).astype(np.float32), camera_info
