import numpy as np
import os
from natsort import natsorted
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

def sorttxtFiles(path):
    """
    The path contains png and txt files, return the matching file names and the calibration path
    Input : path to the txt folders
    Output: Calibration file path, list of matching file names
    """
    data_files = os.listdir(path)
    sorted_files = natsorted([file for file in data_files if file.endswith('.txt')])
    calib_path = os.path.join(path,sorted_files[0])
    matching_files = sorted_files[1:]
    return calib_path, matching_files

def readFiles(file_names,data_path):
    """
    Read the data files matching.txt and for each pair of images return the pixel coordinates and the number of features.
    Input : file_names list of file_names, data_path path to the files
    Output: nFeatures, list of processed data
    """
    nFeatures = []
    length = len(file_names) + 1
    n_lists = int(length*(length-1)/2)
    data_list = [[] for _ in range(n_lists)]
    for i,file_name in enumerate(file_names):
        file_path = os.path.join(data_path,file_name)
        with open(file_path,'r') as file:
            lines = file.readlines()
        # read the total number of feature matches from the first line
        nf = int(lines[0].strip().split(':')[1].strip())
        nFeatures.append(nf)
        rows = []
        #traverse through each line
        for line in lines[1:]:
            line = [item for item in line.strip().split()]
            #get the number of matches
            n_matches = int(line[0])
            ids = []
            for n in range(n_matches-1):
                coord = 3*n+6
                ids.append(int(line[coord]))
            for k,id in enumerate(ids):
                u_i = line[4]
                v_i = line[5]
                u_id = line[3*k + 7]
                v_id = line[3*k + 8]
                px = [float(u_i),float(v_i),float(u_id),float(v_id)]
                loc = i*length - int(i*(i+1)/2) + id - i - 2
                data_list[loc].append(px)

    idx1 = 1
    idx2 = 2
    for j in range(n_lists):
        path = os.path.join(data_path,"Matches")
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path,"matching" + str(idx1) + str(idx2) + ".txt")
        np.savetxt(path,np.array(data_list[j], dtype=float))
        if(idx2 == 5):
            idx1 += 1
            idx2 = idx1 + 1
        else:
            idx2 += 1
            # rows.append(list(map(float, line.split())))
    
    
    return nFeatures,data_list

def getImgNums(n_imgs,id):
    i = 1
    while(id >= n_imgs-1):
        id = id - n_imgs + 1
        i +=1
        n_imgs -= 1
    j = i + 1 + id
    return i,j

def getMatches(pts1,pts2,idxs,n_imgs,id,path):
    """
    View the inliers and outliers after performing RANSAC
    Inputs: pts1 - Array of pixel location of image 1
            pts2 - Array of pixel locations of image 2
            idxs - The list of indexes of inliers in dl
            n_imgs - The number of images taken for performing Sfm
            id - The current id in the list of matching image pairs
            path - The path where the images are stored
    Output: Stored image showing output of RANSAC
    """
    i = 1
    while(id >= n_imgs-1):
        id = id - n_imgs + 1
        i +=1
        n_imgs -= 1
    j = i + 1 + id
    img1 = cv2.imread(os.path.join(path,str(i) + ".png"))
    img2 = cv2.imread(os.path.join(path,str(j) + ".png"))
    imgs = np.hstack((img1,img2))
    for idx in range(pts1.shape[0]):
        color1 = (0,0,255)
        if idx in idxs:
            color2 = (0,255,0)
            pt1 = (int(pts1[idx,0]), int(pts1[idx,1]))
            pt2 = (int(pts2[idx,0]) + int(img1.shape[1]), int(pts2[idx,1]))
            cv2.line(imgs, pt1, pt2, color2, 1)
            cv2.circle(imgs, pt1, 2, color1, -1)
            cv2.circle(imgs, pt2, 2, color1, -1)
        else:
            color2 = color1
        # pt1 = (int(pts1[idx,0]), int(pts1[idx,1]))
        # pt2 = (int(pts2[idx,0]) + int(img1.shape[1]), int(pts2[idx,1]))
        # cv2.line(imgs, pt1, pt2, color2, 1)
        # cv2.circle(imgs, pt1, 2, color1, -1)
        # cv2.circle(imgs, pt2, 2, color1, -1)
    output_path = os.path.join(path,"RANSAC_Imgs")
    if not os.path.exists:
        os.makedirs(output_path)
    cv2.imwrite(output_path + "pair_" + str(i) + str(j) + ".png",imgs)

def drawlines(img1, img2, lines, pts1, pts2):
    """
    Draw epipolar lines and points on images.
    
    Args:
        img1: Image on which we draw the epilines
        img2: Second image (not modified)
        lines: Epipolar lines corresponding to points in the second image
        pts1: Points in the first image
        pts2: Points in the second image
        
    Returns:
        img1_with_lines: First image with epipolar lines
        img2_with_points: Second image with points marked
    """
    r, c = img1.shape[:2]
    img1_with_lines = img1.copy()
    img2_with_points = img2.copy()
    
    # Draw lines on img1
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1_with_lines = cv2.line(img1_with_lines, (x0, y0), (x1, y1), color, 1)
        img1_with_lines = cv2.circle(img1_with_lines, tuple(map(int, pt1)), 5, color, -1)
        img2_with_points = cv2.circle(img2_with_points, tuple(map(int, pt2)), 5, color, -1)
        
    return img1_with_lines, img2_with_points

def visualize_epipolar_lines(img1, img2, pts1, pts2, F, title="Epipolar Lines"):
    """
    Visualize epipolar lines on both images.
    
    Args:
        img1, img2: Input images
        pts1, pts2: Matching points in both images
        F: Fundamental matrix
        title: Title for the plot
    """
    # Ensure points are in the right format
    pts1_reshaped = pts1.reshape(-1, 1, 2).astype(np.float32)
    pts2_reshaped = pts2.reshape(-1, 1, 2).astype(np.float32)
    
    # Find epilines in image 2 corresponding to points in image 1
    lines1 = cv2.computeCorrespondEpilines(pts1_reshaped, 1, F)
    lines1 = lines1.reshape(-1, 3)
    img2_with_lines, img1_with_points = drawlines(img2, img1, lines1, pts2, pts1)
    
    # Find epilines in image 1 corresponding to points in image 2
    lines2 = cv2.computeCorrespondEpilines(pts2_reshaped, 2, F)
    lines2 = lines2.reshape(-1, 3)
    img1_with_lines, img2_with_points = drawlines(img1, img2, lines2, pts1, pts2)
    
    # Create figure with two subplots side by side
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(img1_with_lines)
    plt.title('Image 1 with Epipolar Lines')
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(img2_with_lines)
    plt.title('Image 2 with Epipolar Lines')
    plt.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()