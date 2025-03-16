import argparse
import torch
import os
import time
import cv2
from skimage.metrics import structural_similarity
from NeRFModel import *
from datalogger import Logger
from LoadData import loadDataset
import numpy as np
from tqdm import tqdm
from RenderFuntions import generateBatch, render


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)


def print_memory_usage(tag):
    allocated_memory = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
    reserved_memory = torch.cuda.memory_reserved() / (1024**3)  # Convert to GB
    print(
        f"{tag} - Allocated Memory: {allocated_memory:.2f} GB | Reserved Memory: {reserved_memory:.2f} GB"
    )


def Loss(groundtruth, prediction):
    """
    Input:
        groundtruth: pixel values of image
        prediction : prediction of pixel colors from the neural network
    Output:
        norm_loss: squared norm between groundtruth and prediction error
        psnr     : peak signal to noise ratio"""
    mse2psnr = (
        lambda x: -10.0
        * torch.log(x).to(DEVICE)
        / torch.log(torch.Tensor([10.0])).to(DEVICE)
    )
    mse_loss = ((prediction - groundtruth) ** 2).mean()
    psnr = mse2psnr(mse_loss)
    norm_loss = ((prediction - groundtruth).norm()) ** 2
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
    model = NeRFmodel(
        args.n_pos_freq, args.n_dirc_freq, flag_encoding=args.position_encoding
    ).to(DEVICE)
    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate)
    # setup the scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[5, 10, 15], gamma=0.5
    )
    # initialize a large loss value
    min_loss = float("inf")
    # generate all rays
    rays_total = generateBatch(images, poses, camera_info)
    rays_warm = generateBatch(images[:, 100:300, 100:300, :], poses, camera_info)
    # calculate number of batches

    for epoch in tqdm(range(args.num_epochs)):
        if epoch <= 2:
            rays = rays_warm
        else:
            rays = rays_total

        num_batches = int(len(rays) / args.n_rays_batch)
        for i in tqdm(range(num_batches)):
            # model in training mode for gradients
            model.train()
            # get random indexes
            batch_idxs = np.random.choice(np.arange(len(rays)), args.n_rays_batch)
            # get the batch of rays
            batch_rays = rays[batch_idxs]
            # start time
            start = time.time()
            # get origin batch coordinates
            batch_o = torch.tensor(batch_rays[:, :3]).to(DEVICE)
            # get batch directions
            batch_d = torch.tensor(batch_rays[:, 3:6]).to(DEVICE)
            # get batch image pixel values
            C_r_batch = torch.tensor(batch_rays[:, 6:]).to(DEVICE)
            # predict the image pixel color using NeRFmodel
            C_hat_batch = render(
                model, batch_o, batch_d, args.tn, args.tf, args.n_sample
            )
            # calculate the loss between groundtruth and prediction
            loss, psnr = Loss(C_r_batch, C_hat_batch)
            # clear the gradients from before
            optimizer.zero_grad()
            # caculate the new gradients
            loss.backward()
            optimizer.step()

            del batch_o, batch_d, C_r_batch, C_hat_batch

            if i % 1000 == 0:
                print(
                    f"Iteration: {i}, Train Loss: {loss.item()}, PSNR: {psnr.item()}, Avg time: {time.time()-start}"
                )
                logger.log(
                    tag="train",
                    epoch=epoch,
                    iter=i,
                    loss=loss.item(),
                    psnr=psnr.item(),
                    time=time.time() - start,
                )
            if i % 5000 == 0:
                logger.log(
                    tag="msg", epoch=epoch, iter=i, msg="Performing Validation..."
                )
                val_loss = val(model, epoch, args, mode="val")
                if val_loss < min_loss:
                    min_loss = val_loss
                    logger.log(tag="model_loss", loss=min_loss)
                    torch.save(
                        model.state_dict(),
                        os.path.join(args.checkpoint_path, "best_model.pt"),
                    )

                logger.log(tag="plot")

                logger.log(tag="model", loss=loss.item())
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        args.checkpoint_path,
                        "model_" + str(epoch) + "_" + str(i) + ".pt",
                    ),
                )

        torch.cuda.empty_cache()

        # test the best model
        if (epoch + 1) % 5 == 0:
            test(args, mode="test", epoch=epoch)

        scheduler.step()


def val(model, epoch, args, mode="val"):
    print("---------Validation Mode entered---------")

    print(args.data_path)
    images, poses, camera_info = loadDataset(args.data_path, mode)
    img_idxs = np.arange(len(images))
    rand_idxs = np.random.choice(img_idxs, 4)
    val_imgs = images[rand_idxs]
    val_poses = poses[rand_idxs]

    val_rays = generateBatch(val_imgs, val_poses, camera_info)
    num_batches = int(len(val_rays) / args.n_rays_batch)
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
            print(
                f"Iteration: {i}, Val Loss: {loss.item()}, PSNR: {psnr.item()}, Avg time: {time.time()- val_start}"
            )
            logger.log(
                tag="val",
                epoch=epoch,
                iter=i,
                loss=loss.item(),
                psnr=psnr.item(),
                time=time.time() - val_start,
            )

    avg_loss /= num_batches

    del val_o, val_d, val_C_r, val_C_hat
    torch.cuda.empty_cache()
    return avg_loss


def test(args, mode="test", epoch=0):
    print("---------Test Mode entered---------")
    # Load the test data
    test_images, test_poses, camera_info = loadDataset(args.data_path, mode)
    imgs_path = os.path.join(args.logs_path, "Media", "Epoch" + str(epoch))
    if not os.path.exists(imgs_path):
        os.makedirs(imgs_path)
    image_number = 5
    test_rays = generateBatch(
        np.expand_dims(test_images[image_number], axis=0),
        np.expand_dims(test_poses[image_number], axis=0),
        camera_info,
    )

    model = NeRFmodel(
        embed_pos_L=10, embed_direction_L=4, flag_encoding=args.position_encoding
    ).to(DEVICE)
    model.load_state_dict(
        torch.load(os.path.join(args.checkpoint_path, "best_model.pt"))
    )
    model.eval()
    C_hat_list = []
    psnr_sum = 0.0
    test_start = time.time()
    H, W, _ = test_images[0].shape

    ## Original IMage
    original_img = test_rays[:, 6:] * 255.0
    original_img = original_img.reshape(H, W, 3)

    for i in range(0, len(test_rays), args.n_rays_batch):
        batch = test_rays[i : i + args.n_rays_batch]
        test_o = torch.tensor(batch[:, :3]).to(DEVICE)
        test_d = torch.tensor(batch[:, 3:6]).to(DEVICE)
        test_C_r = torch.tensor(batch[:, 6:]).to(DEVICE)
        test_C_hat = render(model, test_o, test_d, args.tn, args.tf, args.n_sample)
        loss, psnr = Loss(test_C_r, test_C_hat)
        C_hat_list.append(test_C_hat.detach().cpu())
        if psnr > 1e10:
            continue

        psnr_sum += psnr.item()

    img = torch.cat(C_hat_list).numpy().reshape(H, W, 3) * 255.0
    image_bgr = img[..., ::-1]
    image_bgr_original = original_img[..., ::-1]
    psnr = psnr_sum * args.n_rays_batch / len(test_rays)
    print(f"Iteration: {i}, PSNR: {psnr}, Avg time: {time.time()- test_start}")

    cv2.imwrite(
        os.path.join(
            args.logs_path,
            "Media",
            "Nerf_test_" + str(image_number) + "_" + str(epoch) + ".png",
        ),
        image_bgr,
    )
    cv2.imwrite(
        os.path.join(
            args.logs_path,
            "Media",
            "Original_test_" + str(image_number) + "_" + str(epoch) + ".png",
        ),
        image_bgr_original,
    )


def compute_image(image_number, mode, model):

    test_images, test_poses, camera_info = loadDataset(args.data_path, mode)
    test_rays = generateBatch(
        np.expand_dims(test_images[image_number], axis=0),
        np.expand_dims(test_poses[image_number], axis=0),
        camera_info,
    )

    C_hat_list = []
    psnr_sum = 0.0
    test_start = time.time()
    H, W, _ = test_images[0].shape

    ## Original IMage
    original_img = test_rays[:, 6:] * 255.0
    original_img = original_img.reshape(H, W, 3)

    for i in range(0, len(test_rays), args.n_rays_batch):
        batch = test_rays[i : i + args.n_rays_batch]
        test_o = torch.tensor(batch[:, :3]).to(DEVICE)
        test_d = torch.tensor(batch[:, 3:6]).to(DEVICE)
        test_C_r = torch.tensor(batch[:, 6:]).to(DEVICE)
        test_C_hat = render(model, test_o, test_d, args.tn, args.tf, args.n_sample)
        loss, psnr = Loss(test_C_r, test_C_hat)
        C_hat_list.append(test_C_hat.detach().cpu())
        del test_o, test_d, test_C_r, test_C_hat
        if psnr > 1e10:
            continue

        psnr_sum += psnr.item()

    img = torch.cat(C_hat_list).numpy().reshape(H, W, 3) * 255.0
    image_bgr = img[..., ::-1]
    image_bgr_original = original_img[..., ::-1]

    cv2.imwrite(
        os.path.join(
            args.logs_path,
            "Media",
            "Nerf_gif_" + str(image_number) + ".png",
        ),
        image_bgr,
    )
    cv2.imwrite(
        os.path.join(
            args.logs_path,
            "Media",
            "Original_gif_" + str(image_number) + ".png",
        ),
        image_bgr_original,
    )
    return img


def gif_test(args, mode="test"):

    model = NeRFmodel(
        embed_pos_L=10, embed_direction_L=4, flag_encoding=args.position_encoding
    ).to(DEVICE)
    model.load_state_dict(
        torch.load(os.path.join(args.checkpoint_path, "best_model.pt"))
    )
    model.eval()
    images = []
    for k in range(0, 100):
        print("Computing Image", k)
        images.append(compute_image(k, mode, model))
    images = np.array(images)
    print(images.shape)
    images = images.astype(np.uint8)
    path = os.path.join(args.logs_path, "Media", "Nerf_test" + ".gif")
    frame_duration = 0.1  # duration in seconds for each frame
    imageio.imwrite(path, images, duration=frame_duration)


def main(args):
    # Make directories
    if args.position_encoding:
        logs = os.path.join(args.logs_path, args.object + "_Encoding" + "/")
        checkpoint = os.path.join(args.checkpoint_path, args.object + "_Encoding" + "/")
    else:
        logs = os.path.join(args.logs_path, args.object + "_NoEncoding" + "/")
        checkpoint = os.path.join(
            args.checkpoint_path, args.object + "_NoEncoding" + "/"
        )

    if not os.path.exists(logs):
        os.makedirs(logs)

    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)

    # Check CUDA
    print("Running on Deivce: ", DEVICE)
    # load data
    print("Loading data...")
    print("Check arguments")

    # Path with object name
    path = os.path.join(args.data_path, args.object + "/")
    images, poses, camera_info = loadDataset(path, args.mode)

    #    # initialize logger

    global logger
    logger = Logger(logs)
    logger.log(
        tag="args",
        data_path=path,
        log_path=logs,
        mode=args.mode,
        lrate=args.lrate,
        positional_encodings=args.n_pos_freq,
        directional_encodings=args.n_dirc_freq,
        batch_size=args.n_rays_batch,
        samples_per_ray=args.n_sample,
        near_plane_dist=args.tn,
        far_plane_dist=args.tf,
        epochs=args.num_epochs,
        ckpts_path=checkpoint,
    )

    # New arguments
    args.logs_path = logs
    args.data_path = path
    args.checkpoint_path = checkpoint

    print(args.mode)
    print(args.object)
    print(args.n_pos_freq)
    print(args.n_dirc_freq)
    print(args.position_encoding)

    # Section to train or test the Nerf
    if args.mode == "train":
        print("Start training")
        train(images, poses, camera_info, args)
    elif args.mode == "test":
        print("Start testing")
        test(args)
    elif args.mode == "gif":
        print("Start gif")
        gif_test(args)


def configParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./Data", help="dataset path")
    parser.add_argument("--object", default="ship", help="dataset path")
    parser.add_argument("--mode", default="train", help="train | test | val | gif")
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
