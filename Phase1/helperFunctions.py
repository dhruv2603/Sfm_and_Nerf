import numpy as np
import os
from natsort import natsorted
from tqdm import tqdm
import cv2
from matplotlib import pyplot as plt
from functions import plot_banks


def sorttxtFiles(path):
    """
    The path contains png and txt files, return the matching file names and the calibration path
    Input : path to the txt folders
    Output: Calibration file path, list of matching file names
    """
    data_files = os.listdir(path)
    sorted_files = natsorted([file for file in data_files if file.endswith(".txt")])
    calib_path = os.path.join(path, sorted_files[0])
    matching_files = sorted_files[1:]
    return calib_path, matching_files


def readFiles(file_names, data_path):
    """
    Read the data files matching.txt and for each pair of images return the pixel coordinates and the number of features.
    Input : file_names list of file_names, data_path path to the files
    Output: nFeatures, list of processed data
    """
    nFeatures = []
    length = len(file_names) + 1
    n_lists = int(length * (length - 1) / 2)
    data_list = [[] for _ in range(n_lists)]
    for i, file_name in tqdm(enumerate(file_names)):
        file_path = os.path.join(data_path, file_name)
        with open(file_path, "r") as file:
            lines = file.readlines()
        # read the total number of feature matches from the first line
        nf = int(lines[0].strip().split(":")[1].strip())
        nFeatures.append(nf)
        rows = []
        # traverse through each line
        for line in lines[1:]:
            line = [item for item in line.strip().split()]
            # get the number of matches
            n_matches = int(line[0])
            ids = []
            for n in range(n_matches - 1):
                coord = 3 * n + 6
                ids.append(int(line[coord]))
            for k, id in enumerate(ids):
                u_i = line[4]
                v_i = line[5]
                u_id = line[3 * k + 7]
                v_id = line[3 * k + 8]
                px = [u_i, v_i, u_id, v_id]
                loc = i * length - int(i * (i + 1) / 2) + id - i - 2
                data_list[loc].append(px)

    idx1 = 1
    idx2 = 2
    for j in range(n_lists):
        path = os.path.join(data_path, "Matches")
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, "matching" + str(idx1) + str(idx2) + ".txt")
        np.savetxt(path, np.array(data_list[j], dtype=float))
        if idx2 == 5:
            idx1 += 1
            idx2 = idx1 + 1
        else:
            idx2 += 1
            # rows.append(list(map(float, line.split())))

    return nFeatures, data_list


def getMatches(dl, idxs, n_imgs, id, path):
    """
    View the inliers and outliers after performing RANSAC
    Inputs: dl - the list of indexes of corresponding features
            idxs - The list of indexes of inliers in dl
            n_imgs - The number of images taken for performing Sfm
            id - The current id in the list of matching image pairs
            path - The path where the images are stored
    Output: Stored image showing output of RANSAC
    """
    i = 1
    while id >= n_imgs - 1:
        id = id - n_imgs + 1
        i += 1
        n_imgs -= 1
    j = i + 1 + id
    img1 = cv2.imread(os.path.join(path, str(i) + ".png"))
    img2 = cv2.imread(os.path.join(path, str(j) + ".png"))
    imgs = np.hstack((img1, img2))
    for idx in range(len(dl)):
        color1 = (0, 0, 255)
        if idx in idxs:
            color2 = (0, 255, 0)
            pt1 = (int(float(dl[idx][0])), int(float(dl[idx][1])))
            pt2 = (int(float(dl[idx][2])) + int(img1.shape[1]), int(float(dl[idx][3])))
            cv2.line(imgs, pt1, pt2, color2, 1)
            cv2.circle(imgs, pt1, 2, color1, -1)
            cv2.circle(imgs, pt2, 2, color1, -1)
        else:
            None
        output_path = os.path.join(path, "RANSAC_Imgs")
        if not os.path.exists:
            os.makedirs(output_path)
        cv2.imwrite(output_path + "pair_" + str(i) + str(j) + ".png", imgs)


def plotMatches(dl, n_imgs, id, path, projection_1, projection_2, name):
    i = 1
    while id >= n_imgs - 1:
        id = id - n_imgs + 1
        i += 1
        n_imgs -= 1
    j = i + 1 + id
    img1 = cv2.imread(os.path.join(path, str(i) + ".png"))
    img2 = cv2.imread(os.path.join(path, str(j) + ".png"))
    color1 = (0, 0, 255)
    color2 = (0, 255, 0)

    # Plot points original
    for idx in range(len(dl)):
        pt1 = (int(float(dl[idx][0])), int(float(dl[idx][1])))
        pt2 = (int(float(dl[idx][2])), int(float(dl[idx][3])))
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.circle(img1, pt1, 2, color, -1)
        cv2.circle(img2, pt2, 2, color, -1)

    for k in range(0, projection_1.shape[1]):
        projection_image_1 = (int(projection_1[0, k]), int(projection_1[1, k]))
        projection_image_2 = (int(projection_2[0, k]), int(projection_2[1, k]))
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.circle(img1, projection_image_1, 2, color, -1)
        cv2.circle(img2, projection_image_2, 2, color, -1)

    output_path = os.path.join(path, "image_projection")
    os.makedirs(output_path, exist_ok=True)
    cv2.imwrite(output_path + "/" + name + "_" + "image_1" + ".png", img1)
    cv2.imwrite(output_path + "/" + name + "_" + "image_2" + ".png", img2)

    # f, axis = plt.subplots(1, 2, figsize=(15, 8))
    # f.suptitle("Matching Key Points", fontsize="x-large", fontweight="bold", y=0.95)
    # axis[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    # axis[0].set_title("image A Key Points")
    # axis[0].set_axis_off()

    # axis[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    # axis[1].set_title("image B Key Points")
    # axis[1].set_axis_off()
    # plt.show()
    return None


def get_features(n_imgs, id, path):
    i = 1
    while id >= n_imgs - 1:
        id = id - n_imgs + 1
        i += 1
        n_imgs -= 1
    j = i + 1 + id
    img1 = cv2.imread(os.path.join(path, str(i) + ".png"))
    img2 = cv2.imread(os.path.join(path, str(j) + ".png"))

    # Initiate SIFT detector
    sift = cv2.SIFT_create()  # if cv2 version >= 4.4.0
    # sift = cv2.xfeatures2d.SIFT_create() # if cv2 version = 4.3.x

    # Compute SIFT keypoints and descriptors
    kpA, desA = sift.detectAndCompute(img1, None)
    kpB, desB = sift.detectAndCompute(img2, None)

    # FLANN parameters and initialize
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Matching descriptor using KNN algorithm
    matches = flann.knnMatch(desA, desB, k=2)

    # Apply ratio test
    ptsA = []
    ptsB = []

    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:  # was 0.7
            ptsA.append(kpA[m.queryIdx].pt)
            ptsB.append(kpB[m.trainIdx].pt)

    ptsA = np.int32(ptsA)
    ptsB = np.int32(ptsB)
    return ptsA, ptsB


def getMatchesNew(dl, idxs, n_imgs, id, path, name):
    i = 1
    while id >= n_imgs - 1:
        id = id - n_imgs + 1
        i += 1
        n_imgs -= 1
    j = i + 1 + id

    img1 = cv2.imread(os.path.join(path, str(i) + ".png"))
    img2 = cv2.imread(os.path.join(path, str(j) + ".png"))
    imgs = np.hstack((img1, img2))

    color1 = (0, 0, 255)
    color2 = (0, 255, 0)
    for idx in range(len(dl)):
        if idx in idxs:
            pt1 = (int(float(dl[idx][0])), int(float(dl[idx][1])))
            pt2 = (int(float(dl[idx][2])) + int(img1.shape[1]), int(float(dl[idx][3])))
            cv2.line(imgs, pt1, pt2, color2, 1)
            cv2.circle(imgs, pt1, 2, color1, -1)
            cv2.circle(imgs, pt2, 2, color1, -1)

    output_path = os.path.join(path, "images_features")
    os.makedirs(output_path, exist_ok=True)
    cv2.imwrite(
        output_path + "/" + "pair_" + name + "_" + str(i) + str(j) + ".png", imgs
    )
    empty_matrix_pb = np.empty((1, 1), dtype=object)
    imgs_rgb = np.zeros((imgs.shape), dtype=np.uint8)
    imgs_rgb[:, :, 0] = imgs[:, :, 2]
    imgs_rgb[:, :, 2] = imgs[:, :, 0]
    imgs_rgb[:, :, 1] = imgs[:, :, 1]
    empty_matrix_pb[0, 0] = imgs_rgb
    plot_banks(empty_matrix_pb, name, output_path)
    return None
