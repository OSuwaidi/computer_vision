# بسم الله الرحمن الرحيم

import cv2
import numpy as np

######################################################################################################################################################################################################
# Task 1
# Find keypoints in both images and their corresponding feature descriptors (vectors) using SIFT:
# Read the image:
img1 = cv2.imread('box.png', flags=0)  # flags=0 makes it gray scale

# Initiate SIFT detector:
sift = cv2.SIFT_create()

# Find the keypoints:
keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)  # mask=None
print(f"Number of keypoints (1): {len(keypoints_1)}")  # Number of keypoints
print(f"Matrix of feature vectors (1): {descriptors_1.shape} \n")  # Each keypoint is described by a (1x128) feature (row) vector  --> 8 orientations/directions/bins * 16 histogram arrays (histograms derived from dividing a (16x16) local neighborhood around the point into 16 (4x4) grids)

# Display the keypoints:
img_1 = cv2.drawKeypoints(img1, keypoints_1, None)  # Draw keypoints on "img1"
cv2.imshow('box keypoints', img_1)
cv2.imwrite('box_sift.png', img_1)


# Follow same process on image of interest (box) in a scene to apply Matching:
img2 = cv2.imread('box_in_scene.png', 0)

keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)
print(f"Number of keypoints (2): {len(keypoints_2)}")
print(f"Matrix of feature vectors (2): {descriptors_2.shape} \n")

img_2 = cv2.drawKeypoints(img2, keypoints_2, None)
cv2.imshow('box in scene keypoints', img_2)
cv2.imwrite('box_in_scene_sift.png', img_2)
cv2.waitKey(0)
cv2.destroyAllWindows()


######################################################################################################################################################################################################
# Task 2
# Feature Matching between the two pictures:
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)  # Using Brute Force Matcher w/ crossCheck=True to filter out outliers and only return consistent pairs (based on k-nearest-neighbors)  --> The two matches should be each others BEST match
matches = bf.match(descriptors_1, descriptors_2)  # A *list* of matches
print(f"Number of matches w/o Ratio Test: {len(matches)} \n")

# Sorting the matches in ascending order based on their L1 distances:
matches = sorted(matches, key=lambda match: match.distance)  # Sort the list "matches" based on "key"

img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:51], None, flags=2)  # flags=2 only shows "matched" features (hides unmatched keypoints)
cv2.imshow('Matching w/o Ratio Test', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('box_matching.jpg', img3)


######################################################################################################################################################################################################
# Task 3
# Applying Lowe's ratio test for better (more precise) matching results:
bf = cv2.BFMatcher()  # L2 Norm is used by default and crossCheck=False
matches = bf.knnMatch(descriptors_1, descriptors_2, k=2)  # k=2 finds the 2 closest matches for each feature vector/descriptor

good = []
ratio = 0.5  # The lower, the more unique the features are, the better the localization, the better the detection
for match1, match2 in matches:  # We compare the 2 closest matches and see if match1 is "unique enough" (if it's sufficiently different from match2)
    if match1.distance < match2.distance * ratio:
        good.append(match1)

print(f"Number of matches w/ Ratio Test: {len(good)}")
good = sorted(good, key=lambda match: match.distance)

img4 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, good[:51], None, flags=2)
cv2.imshow('Matching w/ Ratio Test', img4)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('box_matching_ratio_test.jpg', img4)


######################################################################################################################################################################################################
# Task 4
# Find keypoints (corners) in both images using Harris Corner Detector:
img1 = cv2.imread('box.png')  # Read colored image in BGR
img1_harris = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
corners = cv2.cornerHarris(img1_harris, 2, 3, 0.04)  # Returns a matrix (same shape as input) that assigns probabilities for each pixel (confidence score) on how probable that it's a corner
trues = (corners > 0.001 * corners.max())  # Apply threshold: only keep corners with a relatively high confidence score
position = np.where(trues == True)
for r, c in zip(position[0], position[1]):
    cv2.circle(img1, (c, r), 1, (0, 255, 255))
cv2.imshow('Corners as "Yellow" in box', img1)

img2 = cv2.imread('box_in_scene.png')
img2_harris = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
corners = cv2.cornerHarris(img2_harris, 2, 3, 0.04)
trues = (corners > 0.001 * corners.max())
position = np.where(trues == True)
for r, c in zip(position[0], position[1]):
    cv2.circle(img2, (c, r), 1, (0, 255, 255))
cv2.imshow('Corners as "Yellow" in box in scene', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
