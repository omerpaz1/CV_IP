import sys
from typing import List

import numpy as np
import cv2
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt

ksize = 5
sigma = 0.3*((ksize-1)*0.5-1)+0.8
kerenl = cv2.getGaussianKernel(ksize,sigma)


def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10, win_size=5) -> (np.ndarray, np.ndarray):
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size:
    :param win_size: The optical flow window size (odd number)
    :return: Original points [[y,x]...], [[dU,dV]...] for each points
    """
    pass


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """
    pyramids = []
    gauss_pyr = gaussianPyr(img,levels)
    for i in range(levels-1):
        temp = gaussExpand(gauss_pyr[i+1],kerenl)
        subt = gauss_pyr[i]-temp
        pyramids.append(subt)
    pyramids.append(gauss_pyr[levels-1])

    return pyramids


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Resotrs the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    maxLevels = len(lap_pyr)
    output = lap_pyr[maxLevels-1]
    for i in range(1,maxLevels):
        print(f'lap_pyr[maxLevels-i-1] shape = {lap_pyr[maxLevels-i-1].shape}')
        output = gaussExpand(output,kerenl) + lap_pyr[maxLevels-i-1]
        print(f'output shape = {output.shape}')
    return output



def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
    """
    output = [img]
    if levels == 0:
        return output

    for i in range(1, levels + 1):
        output.append(reduce(output[i - 1]))

    return output


def gaussExpand(img: np.ndarray, gs_k: np.ndarray) -> np.ndarray:
    """
    Expands a Gaussian pyramid level one step up
    :param img: Pyramid image at a certain level
    :param gs_k: The kernel to use in expanding
    :return: The expanded level
    """
    ker = 2*gs_k
    h,w = img.shape[:2]
    res = np.zeros((h*2,w*2))
    for i in range(h):
        for j in range(w):
            res[i*2,j*2] = img[i,j]
    res = cv2.filter2D(res,-1,ker,cv2)
    res = np.transpose(res)
    res = cv2.filter2D(res,-1,ker)
    res = np.transpose(res)
    return res


def pyrBlend(img_1: np.ndarray, img_2: np.ndarray, mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    """
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: Blended Image
    """
    blended_pyr = []
    laplPyrWhite = laplaceianReduce(img_1,levels)
    laplPyrBlack = laplaceianReduce(img_2,levels)
    gaussPyrMask = gaussianPyr(mask,levels)
    # WRITE YOUR CODE HERE.
    for i in range(0, len(gaussPyrMask)):
        blendedLayer = gaussPyrMask[i]*laplPyrWhite[i] \
                    + (1 - gaussPyrMask[i])*laplPyrBlack[i]
        blended_pyr.append(blendedLayer)
        plt.imshow(blendedLayer)
        plt.show()
    return blended_pyr
    # END OF FUNCTION.



def reduce(image):
  kernel = cv2.getGaussianKernel(ksize=ksize,sigma=sigma)
  convolved = cv2.filter2D(image,-1,kernel,borderType=cv2.BORDER_REPLICATE)
  return convolved[::2, ::2]
  
  '''
  def reduce(image):
  kernel = cv2.getGaussianKernel(ksize=ksize,sigma=sigma)
  convolved = cv2.filter2D(image,-1,kernel,borderType=cv2.BORDER_REPLICATE)
  return convolved[::2, ::2]
  '''



