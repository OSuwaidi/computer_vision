import cv2
from PIL import Image
import imageio
from matplotlib import pyplot as plt
import torch
import numpy as np

# Note: when reading matrices/tensors/arrays, "[[ ]]" ==> 2 dimensions, "[[[ ]]]" ==> 3 dimensions and so on.
# Note: images of data type 'uint8' range from 0 to 255, while images of data type 'float' range from -1 to 1 (that's why displaying any 'float' type image *without normalizing* will look completely white)

im1 = cv2.imread("Cat.jpg")  # "imread" ==> spits out the image as an array of numbers representing pixel values --> *** cv2 reads image in BGR format ***
im2 = Image.open("Cat.jpg")  # "open" ==> opens the image as JPG format (no read = no numbers)
im3 = imageio.imread("Cat.jpg")  # "imread" --> *** imageio reads image in RGB format ***

im1 = im1[:, :, ::-1]  # Convert image from BGR to RGB  (Inverts the columns of the matrix)
# Or: im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)

print(f"im1 = {im1} \nim2 = {im2} \nim3 = {im3} \n\n")
# *** Note ***
# When using "imread", the image is read from (top) left to right, pixel by pixel. And each pixel is represented by 3 colors/columns (RGB).
# Each row represents a line of individual pixels (across the columns) in the image, and each pixel is represented as a 3D row vector --> (#columns x 3) matrix for the first line (row).
# *** When printing the image itself from "imread", it prints out as (RxCx3) (channels=#rows, rows=#columns, columns=3) matrix!!! ***
# As you move down the channels of that matrix, you are moving to the row below in the image. And as you move down the rows of the matrix (in some channel), you are moving to the pixel on the right of the previous pixel.
# Each channel represents a single horizontal line (row) in the image, i.e: the image is divided into "row" number of horizontal lines ==> "row" = number of channels

print(f"im1 shape = {im1.shape} \nim2 shape = {im2.size} \nim3 shape = {im3.shape} \n\n")

plt.figure(figsize=(15, 4))  # figsize=(length, width)
plt.subplot(121)
plt.imshow(im1)
plt.title("CV img")

plt.subplot(122)
plt.imshow(im3)
plt.title("Imagio img")
plt.show()


print(f"im1 red = {im1[:, :, 0]} \nred shape = {im1[:, :, 0].shape} \nim3 red = {im3[:, :, 0]} \n\n")  # *Recall*: array[channel, row, column] ==> [:, :, 0] = take all channels, take all rows, take the first column
# This results in: taking all the rows from the first column of every channel, and since the first column represents the *red* color, it produces a matrix of 'red' values
# To illustrate this:
a = torch.rand(3, 5, 3)  # 3 channels of a (5x3) matrix  --> 3 columns represent RGB
print(f"a = {a} \n\na_red = {a[:, :, 0]}\n\n")  # Produces a matrix (row x column)

# *** However: ***
# matrix "im1[:, :, 0]" wont produce a red image, as that matrix is a (row x column) matrix, and ".imshow()" expects images to have the form: (channels, rows, columns)
# And channels are supposed to represent the number of rows (horizontal lines) in the image, and rows = number of columns, and columns = 3 for RGB
# Therefore, to produce a red image, we have to set the other 2 columns to 0, the ones representing green and blue as such:
im1[:, :, 1:] = 0  # Take all row and all column pixel values from the 2nd to last channels and set them to 0

plt.imshow(im1)
plt.title('Red')
plt.show()


plt.figure(figsize=(15, 4))
plt.subplot(121)
diff = im1 - im3  # We get a "cyany" image because we removed the red part of the image, and since im1 has 0 for green and blue, im3 green and blue values took over
                  # eg: 0 - 240 = + 240 (pixel values reset)
plt.imshow(diff)
plt.title('Without cap')

plt.subplot(122)
im1 = np.float32(im1)  # floating the image sets a cap on the image so that pixel values wont reset when exceeding 255 or 0
diff = im1 - im3  # Now we get a black image since pixel values wont reset even if we go below 0
plt.imshow(diff)
plt.title('With cap')
plt.show()


# Normalizing example:
img = cv2.imread('snake.jpg')
img = np.float32(img)
img /= 255
b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
gray = (b + g + r) / 3


# Alpha (step) Transformation:
alpha = gray.copy()
for r in range(len(img)):
    for c in range(img.shape[1]):
        if alpha[r, c] < 0.5:
            alpha[r, c] = 0
        else:
            alpha[r, c] = 1

cv2.imshow('alpha snake', alpha)
cv2.waitKey(0)


# Negative Transformation:
'''
Classic way:
negative = gray.copy()
for r in range(len(img)):
    for μ in range(img.shape[1]):
        negative[r, μ] = np.exp(-5*negative[r, μ])
'''
negative = gray.copy()
negative = np.exp(-5*negative)  # gamma=5, if (gamma>5) ==> more contrast

cv2.imshow('negative snake', negative)
cv2.waitKey(0)


# Gamma Transformation:
# Method 1:
bright = gray.copy()
bright **= 0.5  # gamma=0.5, if (gamma > 1) --> darker image; if (gamma < 1) --> brighter image

cv2.imshow('bright snake', bright)
cv2.waitKey(0)

# Method 2: (only for darkening)
dark = gray.copy()
dark = np.exp(5 * dark - 5)  # gamma=5, the greater the gamma, the greater the darkening effect

cv2.imshow('dark snake', dark)
cv2.waitKey(0)
