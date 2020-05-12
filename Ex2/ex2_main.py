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
    Thresh = 0.5
    img_path = 'boxman.jpg'
    Orignal_img = cv2.imread(img_path,0)
    #save the original shape of the image.
    # ---------------------------- fun 1 ----------------------------:
    # kernel = np.array([1,2,3])
    # signal_1D = np.array([0,0,2,3,4,3,1,3,6,74,5])
    # img_id = conv1D(signal_1D,kernel)
    # test = np.convolve(signal_1D,kernel)
    # print(img_id)
    # print(test)

    # kernel_size = 5
    # kernel = np.ones((kernel_size,kernel_size))
    # kernel = kernel/np.sum(kernel)
    # myans = conv2D(Orignal_img,kernel)
    # CvAnsImg = cv2.filter2D(Orignal_img,-1,kernel,borderType=cv2.BORDER_REPLICATE) #need to be 'full'
    # diff = np.max(CvAnsImg-myans)
    # print(diff)
    # printByName([myans,CvAnsImg])
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

    # der_x,der_y,magnitude,directions = convDerivative(Orignal_img)
    # printByName([der_x,der_y,magnitude,directions])

    # ---------------------------- fun 3 ----------------------------:
    MyImpSobol, OpenCvSobol = edgeDetectionSobel(Orignal_img,Thresh)
    # # zero_cross_img = edgeDetectionZeroCrossingSimple(Orignal_img)
    # printByName([MyImpSobol, OpenCvSobol])
    f, ax = plt.subplots(1, 2)
    ax[0].imshow(MyImpSobol,'gray')
    ax[1].imshow(OpenCvSobol,'gray')
    plt.show()
    # NMS = edgeDetectionCanny(Orignal_img,100,200)
    # cany = cv2.Canny(Orignal_img,100,200)
    # plt.imshow(cany,cmap='gray')
    # plt.show()
    # gray_image = cv2.GaussianBlur(Orignal_img,(5,5),0)
    # cany = HoughLines(Orignal_img,d)
    # printByName([cany[0],cany[1]])

if __name__ == '__main__':
    main()