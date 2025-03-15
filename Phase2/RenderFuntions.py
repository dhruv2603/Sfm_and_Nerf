import torch
from skimage.metrics import structural_similarity
from NeRFModel import *
import numpy as np


def PixelToRay(images, pose, K):
    """
    Input:
        images: list of all images in the Batch
        pose  : Array of camera poses in world frame
        K     : Intrinsic camera matrix
    Outputs:
        ray origin and direction
    """
    Batch_size, H, W, _ = images.shape
    o = np.zeros((Batch_size, H * W, 3))
    d = np.zeros((Batch_size, H * W, 3))
    values = images.reshape((Batch_size, H * W, -1))

    for i in range(Batch_size):
        cam2world = pose[i]
        u = np.arange(W)
        v = np.arange(H)
        u, v = np.meshgrid(u, v)
        dir = np.stack(
            (
                (u - K[0][2]) / K[0][0],
                -(v - H / 2) / K[1][1],
                -np.ones_like(u),
            ),
            axis=-1,
        )
        dir = (cam2world[:3, :3] @ dir[..., None]).squeeze(-1)
        dir = dir / np.linalg.norm(dir, axis=-1, keepdims=True)

        # Compute Direction
        d[i] = dir.reshape(-1, 3)

        # Compute new Origin
        o[i] = cam2world[:3, 3]

    return o.reshape(-1, 3), d.reshape(-1, 3), values.reshape(-1, 3)


def generateBatch(images, poses, camera_info):
    """
    Input:
        images: all images in dataset
        poses: corresponding camera pose in world frame
        camera_info: image width, height, camera matrix
        shuffle: shuffle the dataset
    Outputs:
        A set of rays
    """
    o, d, values = PixelToRay(images, poses, camera_info["camera_matrix"])
    # shape of rays is (N,9)
    rays = np.concatenate((o, d, values), -1)
    return rays


def render(model, rays_origin, rays_direction, tn=2, tf=6, samples=192, clear_bg=True):
    """
    Input:
        model: NeRF model
        rays_origin: origins of input rays
        rays_direction: direction of input rays
        tn: near plane position
        tf: far plane position
        samples: number of rays in batch
        clear_bg: bool for clear background
    Outputs:
        rgb values of input rays
    """
    # calculate the lower bound of each interval
    lbounds = torch.tensor(
        [tn + (i - 1) * (tf - tn) / samples for i in range(1, samples + 1)]
    )
    # calculate the upper bound of each interval
    ubounds = torch.tensor(
        [tn + i * (tf - tn) / samples for i in range(1, samples + 1)]
    )
    # calculate the distance between each sample
    t_i = (torch.rand(1) * ((tf - tn) / samples) * (ubounds - lbounds) + lbounds).to(
        DEVICE
    )

    delta = torch.cat((torch.diff(t_i), torch.tensor([1e10], device=DEVICE)), -1)
    # calculate the sampled point coords on the ray
    sampled_ray_pts = rays_origin.unsqueeze(1) + t_i.unsqueeze(
        -1
    ) * rays_direction.unsqueeze(1)
    # get the colour and opacity values for the points
    sigma, C_hat = model(
        sampled_ray_pts.reshape(-1, 3),
        rays_direction.expand(samples, sampled_ray_pts.shape[0], 3)
        .transpose(0, 1)
        .reshape(-1, 3),
    )
    C_hat = C_hat.view(sampled_ray_pts.shape[0], samples, 3)
    sigma = sigma.view(sampled_ray_pts.shape[0], samples)
    alpha = 1 - torch.exp(-sigma * delta)

    # calculate the transmission values
    T = torch.cumprod(1 - alpha, dim=1)

    # calculate the importance weights of each sampled point
    weights = torch.cat(
        (torch.ones(T.shape[0], 1, device=T.device), T[:, :-1]), dim=-1
    ).unsqueeze(2) * alpha.unsqueeze(2)

    if clear_bg:
        C_r = (weights * C_hat).sum(1)
        weights_sum = weights.sum(dim=[1, 2])
        return C_r + (1 - weights_sum).unsqueeze(-1)
    else:
        return (weights.unsqueeze(-1) * C_hat).sum(1)
