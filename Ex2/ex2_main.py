import numpy as np
import matplotlib.pyplot as plt
import cv2
from ex2_utils import *


def main():
    print("ID:", myID())
    img_path = 'coins.jpg'
    Orignal_img = cv2.imread(img_path,0)
    #save the original shape of the image.
    # ---------------------------- fun 1 ----------------------------:
    # kernel_size = 15
    # kernel = np.ones((kernel_size,kernel_size))
    # kernel = kernel/kernel.sum()
    # print(len(kernel))
    # myans = conv2D(Orignal_img,kernel)
    # CvAnsImg = cv2.filter2D(Orignal_img,-1,kernel,borderType=cv2.BORDER_REPLICATE) #need to be 'full'
    # print(myans)
    # print(CvAnsImg)
    # print(np.array_equal(myans, CvAnsImg))
    # f, ax = plt.subplots(1,2)

    # ax[0].imshow(myans,cmap='gray')
    # ax[1].imshow(CvAnsImg,cmap='gray')
    # plt.show()
    # ---------------------------- fun 2 ----------------------------:
    # img1DArray = Orignal_img.reshape(-1)
    # imgMatArray = img1DArray.reshape(Orignal_img.shape)
    # npAns = np.convolve(np.array([1,2,3,4,5]),np.array([1,2,3]),'same')
    # myAns = conv1D(np.array([1,2,3,4,5]),np.array([1,2,3]))
    # print(f'myAns = {myAns}')
    # print(f'numpyAns = {npAns}')
    # print(f'equal? {np.array_equal(myAns, npAns)}')
    # print(myans)
    # print('******')
    # print(CvAnsImg)

    convDerivative(Orignal_img)

if __name__ == '__main__':
    main()