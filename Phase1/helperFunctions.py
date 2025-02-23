import numpy as np
import os
from natsort import natsorted
from tqdm import tqdm
import cv2

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
    for i,file_name in tqdm(enumerate(file_names)):
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
                px = [u_i,v_i,u_id,v_id]
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

def getMatches(dl,idxs,n_imgs,id,path):
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
    while(id >= n_imgs-1):
        id = id - n_imgs + 1
        i +=1
        n_imgs -= 1
    j = i + 1 + id
    img1 = cv2.imread(os.path.join(path,str(i) + ".png"))
    img2 = cv2.imread(os.path.join(path,str(j) + ".png"))
    imgs = np.hstack((img1,img2))
    for idx in range(len(dl)):
        color1 = (0,0,255)
        if idx in idxs:
            color2 = (0,255,0)
        else:
            color2 = color1
        pt1 = (int(float(dl[idx][0])), int(float(dl[idx][1])))
        pt2 = (int(float(dl[idx][2])) + int(img1.shape[1]), int(float(dl[idx][3])))
        cv2.line(imgs, pt1, pt2, color2, 1)
        cv2.circle(imgs, pt1, 2, color1, -1)
        cv2.circle(imgs, pt2, 2, color1, -1)
        output_path = os.path.join(path,"RANSAC_Imgs")
        if not os.path.exists:
            os.makedirs(output_path)
        cv2.imwrite(output_path + "pair_" + str(i) + str(j) + ".png",imgs)