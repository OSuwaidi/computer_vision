# بسم الله الرحمن الرحيم

import cv2

# Task 3
# Applying Lowe's ratio test for better (more precise) matching results:
img1 = cv2.imread('box.png', 0)
img2 = cv2.imread('box_in_scene.png', 0)
sift = cv2.SIFT_create()

kp1, desc1 = sift.detectAndCompute(img1, None)
kp2, desc2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher()  # L2 Norm is used by default and crossCheck=False
matches = bf.knnMatch(desc1, desc2, k=2)  # k=2 finds the 2 closest matches for each feature vector/descriptor

good_high = []  # Using a high ratio of 0.9
good_med = []  # Using a medium ratio of 0.6
good_low = []  # Using a low ratio of 0.4

ratios = [0.9, 0.6, 0.4]
for r in ratios:
    for match1, match2 in matches:  # We compare the 2 closest matches and see if match1 is "unique enough" (if it's sufficiently different than match2)
        if match1.distance < match2.distance * r and r == 0.9:
            good_high.append(match1)
        elif match1.distance < match2.distance * r and r == 0.6:
            good_med.append(match1)
        elif match1.distance < match2.distance * r and r == 0.4:
            good_low.append(match1)

print(f"Number of matches w/ 0.9 Ratio: {len(good_high)}")  # Has highest number of matches due to relatively poor criteria for matching
print(f"Number of matches w/ 0.6 Ratio: {len(good_med)}")  # Has significantly less number of matches than ratio=0.9, because somewhat more unique keypoints are chosen
print(f"Number of matches w/ 0.4 Ratio: {len(good_low)}")  # Has the least number of matches due to "strict" criteria which produces only the best (most unique) matches
good_high = sorted(good_high, key=lambda match: match.distance)
good_med = sorted(good_med, key=lambda match: match.distance)
good_low = sorted(good_low, key=lambda match: match.distance)

img_high = cv2.drawMatches(img1, kp1, img2, kp2, good_high[:60], None, flags=2)  # Has the potential to be robust against occlusion and some noise
cv2.imshow('Matching w/ Ratio = 0.9', img_high)

img_med = cv2.drawMatches(img1, kp1, img2, kp2, good_med[:60], None, flags=2)  # Has a "good" balance between the high and low ratios
cv2.imshow('Matching w/ Ratio = 0.6', img_med)

img_low = cv2.drawMatches(img1, kp1, img2, kp2, good_low[:60], None, flags=2)  # Since it has the most unique features, it has better localization and better detection, and is therefore more robust to position variation/translation
cv2.imshow('Matching w/ Ratio = 0.4', img_low)

cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('box_matching_0.9.jpg', img_high)
cv2.imwrite('box_matching_0.6.jpg', img_med)
cv2.imwrite('box_matching_0.4.jpg', img_low)
