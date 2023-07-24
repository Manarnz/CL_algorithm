import cv2 as cv
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
mpl.rc('image', cmap='gray')
import math

def imshow(img):
    plt.imshow(img)
    plt.axis('off')
    plt.show()

# Resizing image array using interpolation
def discretize(matrix, n):
  IMG_HEIGHT = n   #Matrix dimensions in terms of n, number of discretized spaces on image
  IMG_WIDTH = n
  resizedImgArray = cv2.resize(
    imgArray, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA
  )
def findDelta(n):
  size=1
  return size/n #Luca said we should find delta in terms of a given n, and for the time being this is the formula. the size is the original size of the image.

def heat_kernel(kernel_size, delta):
    """Create a kernel using heat equation with input size and deviation"""
    # Create kernel maxtrix with input size
    kernel = np.zeros((kernel_size, kernel_size))

    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i, j] = np.exp(-(i**2 + j**2)/4*delta)/(4*np.pi*delta**2)

    return kernel


def heat_kernel_convolution(image, kernel):
    """Compute heat kernel convolution on input image"""
    # Apply kernel convolution to image
    heat_convoluted = cv.filter2D(image, -1, kernel=kernel)
    heat_convoluted = heat_convoluted.astype(float)

    # Set values above 0.5 to 1 and below to 0
    heat_convoluted[heat_convoluted > 0.1] = 1
    heat_convoluted[heat_convoluted <= 0.1] = 0

    return heat_convoluted

def heat_diffusion(image, kernel, lapse=10):
  """Apply heat diffusion on image over a period"""
  images = [image]
  for i in range(1,lapse):
    conv_img =  heat_kernel_convolution(images[i-1], kernel)
    #append the new convoluted image 
    images.append(conv_img)
  return images

img = cv.imread("meltingice_2.jpg", 0)
imshow(img)
#TEST
kernel = heat_kernel(9, 0.1)
images = heat_diffusion(img, kernel,10)
