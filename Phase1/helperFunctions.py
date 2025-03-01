import numpy as np
import os
from natsort import natsorted
from tqdm import tqdm
import cv2
from matplotlib import pyplot as plt

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

def SetData(dl, K):
    """
    Input : (N,4) size datalist of corresponding images
    Output: (3,N) size of uv image arrays for both images
    """
    sz = len(dl)
    X = np.ones((3, sz))
    U = np.ones((3, sz))

    X_c = np.ones((3, sz))
    U_c = np.ones((3, sz))
    K_inv = np.linalg.inv(K)
    for k in range(sz):
        X[0, k] = float(dl[k][0])
        X[1, k] = float(dl[k][1])

        U[0, k] = float(dl[k][2])
        U[1, k] = float(dl[k][3])

        # Points respect to the center of the image
        X_c[:, k] = K_inv @ X[:, k]
        U_c[:, k] = K_inv @ U[:, k]
    return X, U, X_c, U_c

def homography_RANSAC(pixels1, pixels2, N = 2000, tau = 10):
    best_inliers = []
    
    print("Running RANSAC Iterations for Homography")
    for i in tqdm(range(N)):
        # Randomly select 4 points
        idx = np.random.choice(len(pixels1), 4, replace=False)
        rand_pixels_1 = pixels1[idx]
        rand_pixels_2 = pixels2[idx]
        
        
        rand_pixels_1 = np.float32([rand_pixels_1]).reshape(-1, 1, 2)
        rand_pixels_2 = np.float32([rand_pixels_2]).reshape(-1, 1, 2)
        
        # Compute the Perspective Transform
        H = cv2.getPerspectiveTransform(rand_pixels_1, rand_pixels_2)
        
        # Compute the inliers
        inliers = []
        for j, (pt1, pt2) in enumerate(zip(pixels1, pixels2)):
            pt1 = np.append(pt1, 1)
            pt2 = np.append(pt2, 1)
            est_pt2 = H @ pt1
            est_pt2 = est_pt2 / est_pt2[-1]
            e = np.linalg.norm(pt2 - est_pt2)
            if e < tau:
                inliers.append(j)
        
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
    
    return best_inliers

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
    j = i + 1 + int(id)
    img1 = cv2.imread(os.path.join(path, str(int(i)) + ".png"))
    img2 = cv2.imread(os.path.join(path, str(int(j)) + ".png"))
    imgs = np.hstack((img1, img2))
    for idx in tqdm(range(len(dl))):
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

def get_epipoles(F):
    """
    Get the epipoles
    Input : F  - the fundamental matrix
    Output: e1 - Epipole of image 1
            e2 - Epipole of image 2
    """
    U,_,V = np.linalg.svd(F)
    e1 = V[-1, :]
    e1 = e1/e1[-1]
    
    U,_,V = np.linalg.svd(F.T)
    e2 = V[-1, :]
    e2 = e2/e2[-1]
    
    return e1, e2

def get_epipolar_lines(F, pixels_1, pixels_2):
    """
    Get the epipolar lines
    Inputs: F        - Fundamental matrix
            pixels_1 - (N,2) array of image 1 coordinates
            pixels_2 - (N,2) array of image 2 coordinates
    Output: lines1,lines2
    """
    # Convert to Homogeneous Coordinates
    pixels_1 = np.hstack((pixels_1, np.ones((pixels_1.shape[0], 1))))
    pixels_2 = np.hstack((pixels_2, np.ones((pixels_2.shape[0], 1))))
    lines1 = pixels_1 @ F
    lines2 = pixels_2 @ F.T
    return lines1, lines2

def drawlines(img1, img2, lines, pts1, pts2,path):
    """
    Draw the epipolar lines and corresponding points on the images.
    Inputs: img1
            img2
            lines Array of epipolar lines.
            pts1 (Nx2) matrix of points from the first image.
            pts2 (Nx2) matrix of points from the second image.
    """
    if len(img1.shape) == 2:
        r, c = img1.shape
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    else:
        r, c, _ = img1.shape
        img1 = img1
        img2 = img2
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1] ])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        pt1 = tuple(map(int, pt1))
        pt2 = tuple(map(int, pt2))
        img1 = cv2.circle(img1, tuple(pt1), 2, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 2, color, -1)
    return img1, img2

def plotMatches(dl, idxs, n_imgs, id, path, projection_1, projection_2, name):
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
        if idx in idxs:
            pt1 = (int(float(dl[idx][0])), int(float(dl[idx][1])))
            pt2 = (int(float(dl[idx][2])), int(float(dl[idx][3])))
            cv2.circle(img1, pt1, 2, color1, -1)
            cv2.circle(img2, pt2, 2, color1, -1)

    for k in range(0, projection_1.shape[1]):
        projection_image_1 = (int(projection_1[0, k]), int(projection_1[1, k]))
        projection_image_2 = (int(projection_2[0, k]), int(projection_2[1, k]))
        cv2.circle(img1, projection_image_1, 2, color2, -1)
        cv2.circle(img2, projection_image_2, 2, color2, -1)

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
    return None

def get_idx(i,j):
    """
    From image numbers i and j, return the location of the corresponding list in the list of lists
    Inputs: i,j
    Output: idx
    """
    return (j - 1) * (10 - j) / 2 + i - j - 1

def checkNewFeatures(
        uv_i,
        uv_j,
        master_list,
        i,j,
        X_i,
        x_i,
        needs_triangulation_idxs_list
        ):
    for a, each_row in enumerate(uv_j):
        # Flag to check if the point is already added in the master list
        flag_a_in_ml = 0
        # and each row in Master list
        for each_Mrow in master_list:
            # calculate the length of the Master list row
            Mrow_len = len(each_Mrow)
            k = 0
            # traverse throgh all ids in the row and check if the row has the id j
            while 3 + 3 * k + 1 < Mrow_len:
                if each_Mrow[3 + 3 * k + 1] == j:
                    if (
                        each_Mrow[3 + 3 * k + 2] == each_row[0]
                        and each_Mrow[3 + 3 * k + 3] == each_row[1]
                    ):
                        flag_a_in_ml = 1
                        m = k + 1
                        flag = 0
                        while 3 + 3 * m + 1 < Mrow_len:
                            if each_Mrow[3 + 3 * k + 1] == j:
                                flag = 1
                                break
                            m = m + 1
                        if flag == 1:
                            break
                        each_Mrow.append(i)
                        each_Mrow.append(uv_i[a, 0])
                        each_Mrow.append(uv_i[a, 1])
                        X_i = np.vstack([X_i, each_Mrow[:3]])
                        x_i = np.vstack([x_i, uv_i[a]])
                        break
                k = k + 1
        if flag_a_in_ml == 0:
            # store the index list in the matching list for which there is no world point
            needs_triangulation_idxs_list.append(a)
    return X_i,x_i,master_list, needs_triangulation_idxs_list