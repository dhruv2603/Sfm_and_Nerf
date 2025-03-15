#!/usr/bin/env python3
import argparse
import os
import time
import cv2
from LoadData import loadDataset
import numpy as np
import rclpy
from rclpy.node import Node


def main(args):

    # Path with object name
    path = os.path.join(args.data_path, args.object + "/")
    images, poses, camera_info = loadDataset(path, args.mode)
    print(poses.shape)


def configParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./Data", help="dataset path")
    parser.add_argument("--object", default="ship", help="dataset path")
    parser.add_argument("--mode", default="train", help="train | test | val")
    parser.add_argument(
        "--lrate", type=float, default=5e-4, help="training learning rate"
    )
    parser.add_argument(
        "--n_pos_freq",
        type=int,
        default=10,
        help="number of positional encoding frequencies for position",
    )
    parser.add_argument(
        "--n_dirc_freq",
        type=int,
        default=4,
        help="number of positional encoding frequencies for viewing direction",
    )
    parser.add_argument(
        "--n_rays_batch", type=int, default=1024, help="number of rays per batch"
    )
    parser.add_argument(
        "--n_sample", type=int, default=100, help="number of sample per ray"
    )
    parser.add_argument("--tn", type=int, default=2, help="tn Near plane distance")
    parser.add_argument("--tf", type=int, default=6, help="tf Far plane distance")
    parser.add_argument(
        "--num_epochs", type=int, default=20, help="number of epochs for training"
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=10000,
        help="number of max iterations for training",
    )
    parser.add_argument("--logs_path", default="./Logs", help="logs path")
    parser.add_argument(
        "--checkpoint_path", default="./Checkpoints", help="checkpoints path"
    )
    parser.add_argument(
        "--load_checkpoint", default=True, help="whether to load checkpoint or not"
    )
    parser.add_argument(
        "--save_ckpt_iter", default=1000, help="num of iteration to save checkpoint"
    )
    parser.add_argument(
        "--images_path", default="./image/", help="folder to store images"
    )
    parser.add_argument(
        "--position_encoding",
        type=bool,
        default=True,
        help="position_encoding",
    )
    return parser


if __name__ == "__main__":
    parser = configParser()
    args = parser.parse_args()
    main(args)
