# بِسْمِ ٱللَّٰهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ و به نستعين

import numpy as np
from math import log2
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread("frog.png", 0)
# img = img[:, :, ::-1]
# plt.imshow(img, cmap='gray')
# plt.axis("off")
# plt.show()

window_size = int(np.std(img))
# Take the moving average (rolling mean) of the image's flattened intensities (to smoothen out any sharp, peaking intensities, reducing bias)
smoothed_img = np.convolve(img.ravel(), np.ones(window_size)/window_size, mode='same')  # 'valid' ensures no padding


def img_entropy(image: np.ndarray, th: int):  # basically returns the "th" value that is the median of all intensities but in O(n) time
    p_background = (image < th).mean()
    return -(p_background*log2(p_background) + (1-p_background)*log2(1-p_background))


# Using the smoothed_img's extreme points reduces the search space since the rolling mean always increases the min value and decreases the max value
optimal_th = max(range(int(smoothed_img.min())+1, int(smoothed_img.max())), key=lambda th: img_entropy(smoothed_img, th))  # "max" selection is done according to the "key" criteria


max_img = img.copy()
max_img[max_img < optimal_th] = 0
max_img[max_img >= optimal_th] = 255

plt.imshow(max_img, cmap='gray')
plt.axis("off")
plt.tight_layout()
plt.show()
