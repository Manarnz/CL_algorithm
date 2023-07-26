import cv2 as cv
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy import ndimage
import imageio
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animationt
mpl.rc('image', cmap='gray')
import math

def imshow(img):
    plt.imshow(img)
    plt.colorbar()
    # plt.axis('off')
    plt.show()

def time_step(n):
    # image size
    size = 512
    delta = size / n
    return delta

def heat_kernel(kernel_size, delta):
    """Create a kernel using heat equation with input size and deviation"""
    # Create kernel maxtrix with input size
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - center
            y = j - center
            kernel[i, j] = np.exp(-(x**2 + y**2)/4)/(4*np.pi*delta**2)
    kernel /= np.sum(kernel)
    print(kernel)
    return kernel

def heat_kernel_convolution(image, kernel):
    """Compute heat kernel convolution on input image"""
    # Apply kernel convolution to image
    heat_convoluted = ndimage.convolve(image, kernel, mode='reflect')
    heat_convoluted = heat_convoluted.astype(float)

    # Set values above 0.5 to 1 and below to 0
    heat_convoluted[heat_convoluted > 0.3] = 1
    heat_convoluted[heat_convoluted <= 0.3] = 0

    return heat_convoluted

def heat_diffusion(image, kernel, lapse):
  """Apply heat diffusion on image over a period"""
  images = [image]
  # imshow(image)
  for i in range(lapse):
    conv_img =  heat_kernel_convolution(images[i], kernel)
    #append the new convoluted image 
    images.append(conv_img)
  for img in images:
    img *= 255
  imageio.mimsave('circle.gif', images, duration = 50)
  return images

def generate_circle_img():
    img = np.zeros((512,512))
    cc= (256,256)
    radius = 100
    cv.circle(img,cc,radius,(1,0,255),-1)
    return img


#TEST
img = generate_circle_img()
delta = time_step(51200)
kernel = heat_kernel(15, delta)

imshow(img)
img2=heat_kernel_convolution(img,kernel)
imshow(img2)
images = heat_diffusion(img, kernel,300)

#TEST2
# read image
img_disk = cv.imread("disk.jpeg", 0)

# resize the image
IMG_HEIGHT, IMG_WIDTH = 512,512
img_disk = cv.resize(img_disk,(IMG_HEIGHT, IMG_WIDTH))
img_disk = 1-img_disk/255
imshow(img_disk)

delta = time_step(51200)
kernel = heat_kernel(15, delta)

images = heat_diffusion(img_disk, kernel,100)
