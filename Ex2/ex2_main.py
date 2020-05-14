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


def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

def main():
    print("ID:", myID())
    Thresh = 0.5
    img_path = 'ema.jpg'
    Orignal_img = cv2.imread(img_path,0)
    highThreshold = Orignal_img.max() * 0.12
    lowThreshold = highThreshold * 0.03
    singel_Thresh = 0.5
    #save the original shape of the image.
    # ---------------------------- fun 1 ----------------------------:
    # kernel = np.array([1,2,3])
    # signal_1D = np.array([0,0,2,3,4,3,1,3,6,74,5])
    # img_id = conv1D(signal_1D,kernel)
    # test = np.convolve(signal_1D,kernel)
    # print(img_id)
    # print(test)

    # kernel_size = 6
    # kernel = np.ones((kernel_size,kernel_size))
    # kernel = kernel/np.sum(kernel)
    # myans = conv2D(Orignal_img,kernel)
    # CvAnsImg = cv2.filter2D(Orignal_img,-1,kernel,borderType=cv2.BORDER_REPLICATE) #need to be 'full'
    # diff = np.max(CvAnsImg-myans)
    # print(diff)
    # printByName([myans,CvAnsImg])
    # print(mse(myans,CvAnsImg))

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
    # ---------------------------- fun 2.2 ----------------------------:
    # burrimg = blurImage2(Orignal_img,5)
    # burrGar = cv2.GaussianBlur(Orignal_img,(5,5),0)
    # diff = np.max(burrGar-burrimg)
    # print(diff)
    # ---------------------------- fun 3 ----------------------------:
    # MyImpSobol,OpenCvSobol = edgeDetectionSobel(Orignal_img,Thresh)
    # NMS = edgeDetectionCanny(Orignal_img,highThreshold,lowThreshold)
    # print(mse(MyImpSobol,OpenCvSobol))
    # zero_cross = edgeDetectionZeroCrossingSimple(Orignal_img)
    # log = edgeDetectionZeroCrossingLOG(Orignal_img)
    # print(mse(zero_cross,log))
    # print()
    # printByName([MyImpSobol,NMS])

    NMS = edgeDetectionCanny(Orignal_img,highThreshold,lowThreshold)
    canny = cv2.Canny(Orignal_img,0.2,0.03)
    f, ax = plt.subplots(1, 2)
    ax[0].imshow(NMS,'gray')
    ax[1].imshow(canny,'gray')
    plt.show()
    # gray_image = cv2.GaussianBlur(Orignal_img,(5,5),0)
    # cany = HoughLines(Orignal_img,d)
    # printByName([cany[0],cany[1]])

if __name__ == '__main__':
    main()