import cv2
import numpy as np
import os

# Path to the folder containing your images
input_folder = "./val"
# Path to a folder where processed images will be saved
output_folder = "./val_new"

# Create the output folder if it doesn't already exist
os.makedirs(output_folder, exist_ok=True)

# Define a lower and upper range for gray in BGR
# Adjust these values if needed
lower_gray = np.array([120, 120, 120], dtype=np.uint8)
upper_gray = np.array([200, 200, 200], dtype=np.uint8)

# Loop through all files in the input folder
for filename in os.listdir(input_folder):
    # Process only image files (add other extensions if needed)
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        # Read the image
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)

        if img is None:
            # If the image can't be read, skip it
            continue

        # Create a mask where gray pixels are within the defined range
        mask = cv2.inRange(img, lower_gray, upper_gray)

        # Replace all pixels in the mask with white (255,255,255)
        img[mask != 0] = [255, 255, 255]

        # Save the modified image to the output folder
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, img)

print("Processing complete. Modified images are saved in:", output_folder)
