import os
import cv2 as cv
import numpy as np
from scipy import ndimage
import imageio
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
import setting
import heat_kernel
import mnist_dataset

mpl.rc('image', cmap='gray')

def imshow(img):
    plt.imshow(img)
    plt.colorbar()
    # plt.axis('off')
    plt.show()


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
  for i in range(lapse):
    conv_img =  heat_kernel_convolution(images[i], kernel.data)
    #append the new convoluted image 
    images.append(conv_img)
  for img in images:
    img *= 255
  filename = 'dataset/'+str(kernel.size)+'_'+str(kernel.timestep)+'.gif'
  imageio.mimsave(filename, images, duration = 50)
  return images

def generate_circle_img():
    img = np.zeros((512,512))
    num_circles = random.randint(1,4)
    for n in range(num_circles):
      cc = (random.randint(100, 400),random.randint(100, 400))
      radius = random.randint(1, 100)
      cv.circle(img,cc,radius,(1,0,255),-1)
    # imshow(img)
    return img




if __name__ == '__main__':
    count = 30
    path = setting.TRAIN_DATASET_PATH    
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(count):
      img = generate_circle_img()
      kernel_size = 10
      time_step = setting.IMGSIZE*random.randint(1,100)
      kernel = heat_kernel.heat_kernel(time_step, kernel_size)
      images = heat_diffusion(img, kernel,100)