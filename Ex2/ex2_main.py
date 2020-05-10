import numpy as np
import matplotlib.pyplot as plt
import cv2
from ex2_utils import *

def printByName(listImg):
    fig = plt.figure(figsize=(20, 10))
    for i in range(len(listImg)):
        a = fig.add_subplot(2,3,i+1)
        plt.imshow(listImg[i],'gray')
        a.set_title(f'image{[i]}')
    plt.show()



def main():
    print("ID:", myID())
    img_path = 'coins.jpg'
    Orignal_img = cv2.imread(img_path,0)
    #save the original shape of the image.
    # ---------------------------- fun 1 ----------------------------:
    kernel_size = 5
    kernel = np.ones((kernel_size,kernel_size))
    print(kernel)
    kernel = kernel/np.sum(kernel)
    myans = conv2D(Orignal_img,kernel)
    CvAnsImg = cv2.filter2D(Orignal_img,-1,kernel) #need to be 'full'
    diff = np.max(CvAnsImg-myans)
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

    # directions,magnitude,x_der,y_der = convDerivative(Orignal_img)
    # printByName([Orignal_img,directions,magnitude,x_der,y_der])

    # ---------------------------- fun 3 ----------------------------:
    myimp, sobleimp = edgeDetectionSobel(Orignal_img)
    # zero_cross_img = edgeDetectionZeroCrossingSimple(Orignal_img)
    printByName([myimp,sobleimp])
if __name__ == '__main__':
    main()