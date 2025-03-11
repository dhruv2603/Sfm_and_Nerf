import argparse
import glob
from tqdm import tqdm
import random
from torch.utils.tensorboard import SummaryWriter
import imageio
import torch
import matplotlib.pyplot as plt
import os
import json
from natsort import natsorted
import imageio.v3 as imageio
import time
import cv2
import pdb

from NeRFModel import *
from datalogger import Logger

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)

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
    img_path = os.path.join(data_path,mode)
    # Load the json file
    with open(os.path.join(data_path,"transforms_" + mode + ".json"),'r') as f:
        data = json.load(f)
    # Extract the camera angle
    camera_angle_x = data.get("camera_angle_x",0)
    # Load the images
    files = natsorted(os.listdir(img_path))
    # Resize the images
    images = [cv2.resize(imageio.imread(os.path.join(img_path, i)), (400, 400), interpolation=cv2.INTER_LINEAR) for i in files if i.endswith(".png") and "_depth_" not in i]
    # images = [imageio.imread(os.path.join(img_path, i)) for i in files if i.endswith(".png") and "_depth_" not in i]
    images = (np.array(images)/255.).astype(np.float32)
    # Make 3 channel instead of 4 channel
    if images.shape[-1] == 4: # RGBA --> RGB
        images = images[..., :3] * images[..., -1:]  + (1 - images[..., -1:])
    # Specify the image size
    width = images[0].shape[1]
    height = images[0].shape[0]
    # Camera matrix for a pin-hole camera model
    f_x = 0.5*width/np.tan(camera_angle_x*0.5)
    f_y = f_x
    c_x = width/2
    c_y = height/2
    camera_matrix = np.array([[f_x, 0, c_x],
                              [0, f_y, c_y],
                              [0, 0, 1]])
    # Extract the camera poses
    pose = []
    for frame in data["frames"]:
        transform_matrix = np.array(frame["transform_matrix"])
        pose.append(transform_matrix)
        # rotation = transform_matrix[:3,:3]
        # position = transform_matrix[:3,3]
        # pose.append({"R":rotation,"T":position})
    
    camera_info = {"width":width,"height":height,"camera_matrix":camera_matrix}

    return images, np.array(pose).astype(np.float32), camera_info


def PixelToRay(images, pose, K):
    """
    Input:
        images: list of all images in the Batch
        pose  : Array of camera poses in world frame
        K     : Intrinsic camera matrix
    Outputs:
        ray origin and direction
    """
    Batch_size,H,W,_ = images.shape
    o = np.zeros((Batch_size,H*W,3))
    d = np.zeros((Batch_size,H*W,3))
    values = images.reshape((Batch_size,H*W,-1))

    for i in range(Batch_size):
        cam2world = pose[i]
        u = np.arange(W)
        v = np.arange(H)
        u,v = np.meshgrid(u,v)
        dir = np.stack(((u-K[0][2])/K[0][0],-(v - H/2)/K[1][1],-np.ones_like(u)),axis=-1)
        dir = (cam2world[:3,:3]@dir[...,None]).squeeze(-1)
        dir = dir/np.linalg.norm(dir,axis=-1,keepdims=True)
        d[i] = dir.reshape(-1,3)
        o[i] = cam2world[:3,3]
    return o.reshape(-1,3), d.reshape(-1,3), values.reshape(-1,3)

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
    o,d,values = PixelToRay(images,poses,camera_info["camera_matrix"])
    # shape of rays is (N,9)
    rays = np.concatenate((o,d,values),-1)
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
    #calculate the lower bound of each interval
    lbounds = torch.tensor([tn + (i - 1) * (tf - tn) / samples for i in range(1, samples + 1)])
    #calculate the upper bound of each interval
    ubounds = torch.tensor([tn + i * (tf - tn) / samples for i in range(1, samples + 1)])
    #calculate the distance between each sample
    t_i = torch.zeros(samples, device=DEVICE)
    for i in range(samples):
        t_i[i] = torch.rand(1, device=DEVICE)*((tf - tn) / samples) * (ubounds[i] - lbounds[i]) + lbounds[i]
    delta = torch.cat((torch.diff(t_i),torch.tensor([1e10],device=DEVICE)),-1)
    #calculate the sampled point coords on the ray
    sampled_ray_pts = rays_origin.unsqueeze(1) + t_i.unsqueeze(-1)*rays_direction.unsqueeze(1)
    #get the colour and opacity values for the points
    sigma,C_hat = model(sampled_ray_pts.reshape(-1,3), rays_direction.expand(samples,sampled_ray_pts.shape[0],3).transpose(0,1).reshape(-1,3))
    C_hat   = C_hat.view(sampled_ray_pts.shape[0],samples,3)
    sigma   = sigma.view(sampled_ray_pts.shape[0],samples)
    alpha   = 1 - torch.exp(-sigma*delta)
    #calculate the transmission values
    T       = torch.cumprod(1-alpha, dim = 1)
    #calculate the importance weights of each sampled point
    weights = torch.cat((torch.ones(T.shape[0],1,device = T.device),T[:,:-1]),dim=-1).unsqueeze(2)*alpha.unsqueeze(2)

    if clear_bg:
        C_r = (weights*C_hat).sum(1)
        weights_sum = weights.sum(dim=[1,2])
        return C_r + (1 - weights_sum).unsqueeze(-1)
    else:
        return (weights.unsqueeze(-1)*C_hat).sum(1)


def Loss(groundtruth, prediction):
    """
    Input:
        groundtruth: pixel values of image
        prediction : prediction of pixel colors from the neural network
    Output:
        norm_loss: squared norm between groundtruth and prediction error
        psnr     : peak signal to noise ratio"""
    mse2psnr = lambda x : -10. * torch.log(x).to(DEVICE) / torch.log(torch.Tensor([10.])).to(DEVICE)
    mse_loss = ((prediction - groundtruth)**2).mean()
    psnr = mse2psnr(mse_loss)
    norm_loss = ((prediction - groundtruth).norm())**2
    return norm_loss, psnr


def train(images, poses, camera_info, args):
    """
    Training function for the NeRF model
    Input:
        images     : all the images in the training dataset
        poses      : transformation matrices for carmera pose in world frame
        camera_info: image width, image height, camera instrinsic matrix
    """
    print("-----Training Mode entered -----")
    # Instantiate the model
    model = NeRFmodel(args.n_pos_freq,args.n_dirc_freq).to(DEVICE)
    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lrate)
    # setup the scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,10,15],gamma = 0.5)
    # initialize a large loss value
    min_loss = float('inf')
    # generate all rays
    rays = generateBatch(images,poses,camera_info)
    # calculate number of batches
    num_batches = int(len(rays)/args.n_rays_batch)

    for epoch in tqdm(range(args.num_epochs)):
        for i in tqdm(range(num_batches)):
            # model in training mode for gradients
            model.train()
            # get random indexes
            batch_idxs = np.random.choice(np.arange(len(rays)),args.n_rays_batch)
            # get the batch of rays
            batch_rays = rays[batch_idxs]
            # start time
            start        = time.time()
            # get origin batch coordinates
            batch_o      = torch.tensor(batch_rays[:,:3]).to(DEVICE)
            # get batch directions
            batch_d      = torch.tensor(batch_rays[:,3:6]).to(DEVICE)
            # get batch image pixel values
            C_r_batch = torch.tensor(batch_rays[:,6:]).to(DEVICE)
            # predict the image pixel color using NeRFmodel
            C_hat_batch  = render(model, batch_o, batch_d, args.tn, args.tf, args.n_sample)
            # calculate the loss between groundtruth and prediction
            loss,psnr = Loss(C_r_batch, C_hat_batch)
            # clear the gradients from before
            optimizer.zero_grad()
            # caculate the new gradients
            loss.backward()
            optimizer.step()

            if i % 1000 == 0:
                print(f'Iteration: {i}, Train Loss: {loss.item()}, PSNR: {psnr.item()}, Avg time: {time.time()-start}')
                logger.log(tag='train', epoch=epoch, iter=i, loss=loss.item(), psnr=psnr.item(), time=time.time()-start)
                logger.log(tag='msg', epoch=epoch, iter=i, msg='Performing Validation...')
                val_loss = val(model, epoch, args, mode='val')
                if val_loss < min_loss:
                    min_loss = val_loss
                    logger.log(tag='model_loss', loss=min_loss)
                    # torch.save(model.state_dict(), log_path+'/'+log_dir+'/model/best_model.pt')
                    torch.save(model.state_dict(), os.path.join(args.checkpoint_path,"best_model.pt"))

                logger.log(tag='plot')

                logger.log(tag='model', loss=loss.item())
                # torch.save(model.state_dict(), log_path+'/'+log_dir+'/model/model_'+str(epoch)+'_'+str(i)+'.pt')
                torch.save(model.state_dict(), os.path.join(args.checkpoint_path,"model_" + str(epoch) + "_" + str(i) + ".pt"))

        # test the best model
        if((epoch + 1)%4 == 0):
            test(args.data_path, mode='test', log_dir=args.logs_path, img_name='image_'+str(epoch)+'_'+str(i)+'.png')

        scheduler.step()
    
def val(model, epoch, args, mode = "val"):
    print("---------Validation Mode entered---------")

    images,poses,camera_info = loadDataset(args.data_path, mode)
    img_idxs = np.arange(len(images))
    rand_idxs = np.random.choice(img_idxs, 4)
    val_imgs = images[rand_idxs]
    val_poses = poses[rand_idxs]

    val_rays = generateBatch(val_imgs, val_poses, camera_info)
    num_batches = int(len(val_rays)/args.n_rays_batch)
    avg_loss = 0.0

    for i in range(num_batches):
        val_idxs = np.random.choice(np.arange(len(val_rays)), args.n_rays_batch)
        batch = val_rays[val_idxs]

        val_start = time.time()

        val_o = torch.tensor(batch[:, :3]).to(DEVICE)
        val_d = torch.tensor(batch[:, 3:6]).to(DEVICE)
        val_C_r = torch.tensor(batch[:, 6:]).to(DEVICE)

        val_C_hat = render(model, val_o, val_d, args.tn, args.tf, args.n_sample)
        loss, psnr = Loss(val_C_r, val_C_hat)
        avg_loss += loss.item()

        if i % 1000 == 0:
            print(f'Iteration: {i}, Val Loss: {loss.item()}, PSNR: {psnr.item()}, Avg time: {time.time()- val_start}')
            logger.log(tag='val', epoch=epoch, iter=i, loss=loss.item(), psnr=psnr.item(), time=time.time()-val_start)

    avg_loss /= (num_batches)

    return avg_loss

    
    
def test(model, epoch, args, mode = "test"):
    print("---------Test Mode entered---------")

    test_images,test_poses, camera_info = loadDataset(args.data_path, mode)
    # img_idxs = np.arange(len(images))
    # rand_idxs = np.random.choice(img_idxs, 4)
    # val_imgs = images[rand_idxs]
    # val_poses = poses[rand_idxs]

    test_rays = generateBatch(test_images, test_poses, camera_info)
    num_batches = int(len(test_rays)/args.n_rays_batch)
    avg_loss = 0.0
    count = len(test_rays)
    # for i in range(num_batches):
        # test_idxs = np.random.choice(np.arange(len(test_rays)), args.n_rays_batch)
        # batch = test_rays[test_idxs]
    while count > 0:
        if count > args.n_rays_batch:
            batch = test_rays[len(test_rays) - count:len(test_rays) - count + args.n_rays_batch]
            count = count - args.n_rays_batch
        else:
            batch = test_rays[len(test_rays) - count:]
            count = 0

        test_start = time.time()

        test_o = torch.tensor(batch[:, :3]).to(DEVICE)
        test_d = torch.tensor(batch[:, 3:6]).to(DEVICE)
        test_C_r = torch.tensor(batch[:, 6:]).to(DEVICE)

        test_C_hat = render(model, test_o, test_d, args.tn, args.tf, args.n_sample)
        loss, psnr = Loss(test_C_r, test_C_hat)
        avg_loss += loss.item()

        if i % 1000 == 0:
            print(f'Iteration: {i}, Test Loss: {loss.item()}, PSNR: {psnr.item()}, Avg time: {time.time()- test_start}')
            logger.log(tag='model_loss', loss=loss.item())

    avg_loss /= (num_batches)

    return avg_loss



def main(args):
    # Make directories
    if not os.path.exists(args.logs_path):
        os.makedirs(args.logs_path)
    if not os .path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    # Check CUDA
    print("Running on Deivce: ", DEVICE)
    # load data
    print("Loading data...")
    images, poses, camera_info = loadDataset(args.data_path, args.mode)
    # initialize logger
    global logger
    logger = Logger(args.logs_path)
    logger.log(tag='args',data_path = args.data_path, log_path = args.logs_path, mode = args.mode, lrate=args.lrate, 
               positional_encodings = args.n_pos_freq, directional_encodings = args.n_dirc_freq, batch_size = args.n_rays_batch, 
               samples_per_ray = args.n_sample, near_plane_dist = args.tn, far_plane_dist = args.tf, epochs = args.num_epochs,
               ckpts_path = args.checkpoint_path)

    if args.mode == 'train':
        print("Start training")
        train(images, poses, camera_info, args)
    elif args.mode == 'test':
        print("Start testing")
        args.load_checkpoint = True
        test(images, poses, camera_info, args)

def configParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',default="./Data/lego/",help="dataset path")
    parser.add_argument('--mode',default='train',help="train | test | val")
    parser.add_argument('--lrate',type=float,default=5e-4,help="training learning rate")
    parser.add_argument('--n_pos_freq',type=int,default=10,help="number of positional encoding frequencies for position")
    parser.add_argument('--n_dirc_freq',type=int,default=4,help="number of positional encoding frequencies for viewing direction")
    parser.add_argument('--n_rays_batch',type=int,default=32*32,help="number of rays per batch")
    parser.add_argument('--n_sample',type=int,default=256,help="number of sample per ray")
    parser.add_argument('--tn', type=int, default=2, help='tn Near plane distance')
    parser.add_argument('--tf', type=int, default=6, help='tf Far plane distance')
    parser.add_argument('--num_epochs', type=int, default=20, help="number of epochs for training")
    parser.add_argument('--max_iters',type=int,default=10000,help="number of max iterations for training")
    parser.add_argument('--logs_path',default="./Logs",help="logs path")
    parser.add_argument('--checkpoint_path',default="./Checkpoints",help="checkpoints path")
    parser.add_argument('--load_checkpoint',default=True,help="whether to load checkpoint or not")
    parser.add_argument('--save_ckpt_iter',default=1000,help="num of iteration to save checkpoint")
    parser.add_argument('--images_path', default="./image/",help="folder to store images")
    return parser

if __name__ == "__main__":
    parser = configParser()
    args = parser.parse_args()
    main(args)