import cv2
import numpy as np
from tqdm import tqdm
# Load images
images = [cv2.imread(f"P2Data/{i}.jpeg", cv2.IMREAD_GRAYSCALE) for i in range(1, 7)]
for img in images:
    img = np.resize()
# Initialize SIFT detector
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(images[0], None)
kp1, des2 = sift.detectAndCompute(images[1], None)
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)


# Iterate over each pair of images
# for i in tqdm(range(len(images))):
#     for j in tqdm(range(i + 1, len(images))):

#         # List to store pixel values of matched keypoints for this pair of images
#         matches_list = []

#         # Detect keypoints and compute descriptors for the two images
#         kp1, des1 = sift.detectAndCompute(images[i], None)
#         kp2, des2 = sift.detectAndCompute(images[j], None)

#         # Use BFMatcher to match the descriptors
#         bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
#         matches = bf.match(des1, des2)

        # # Store matched (u, v) coordinates for both images in the list
        # for match in matches:
        #     u1, v1 = kp1[match.queryIdx].pt  # (u, v) from the first image
        #     u2, v2 = kp2[match.trainIdx].pt  # (u, v) from the second image

        #     # Append the coordinates as a tuple (u1, v1, u2, v2)
        #     matches_list.append([u1, v1, u2, v2])

        # # Save the matches to a text file
        # with open(f'custom_matches{i+1}_{j+1}.txt', 'w') as file:
        #     for match in matches_list:
        #         file.write(f"{match[0]},{match[1]},{match[2]},{match[3]}\n")

        # print(f"Matches saved to 'custom_matches{i+1}_{j+1}.txt'")

# Output the last matches list
# print("Matches (u1, v1, u2, v2):", matches_list)

