import numpy as np
import matplotlib.pyplot as plt
import cv2
from ex2_utils import *



def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	# return the MSE, the lower the error, the more "similar"
	return err

def main():
    print("ID:", myID())
    img_path = 'tom.jpg'
    Orignal_img = cv2.imread(img_path,0)
    #save the original shape of the image.
    # ---------------------------- 1 Convolution ----------------------------:
        # 1.1: Convd1D
    # print('---------------------------- 1 Convolution ----------------------------:')
    print('# 1.1: Convd1D')
    kernel_1D = np.array([1,2,3])
    signal_1D = np.array([0,1,2,4,5])
    My_Conv1D_Img = conv1D(signal_1D, kernel_1D)
    CV2_Conv1D_Img = np.convolve(signal_1D, kernel_1D,'full')
    print(f'My_Conv1D_Img ={My_Conv1D_Img}')
    print(f'CV2_Conv1D_Img ={CV2_Conv1D_Img}')
    print(f'My_Conv1D_Img and CV2_Conv1D_Img equal? {np.array_equal(My_Conv1D_Img,CV2_Conv1D_Img)}')
        # 1.2: Convd2D
    print('# 1.2: Convd2D')
    kernel_size = 5
    kernel = np.ones((kernel_size,kernel_size))
    kernel_2D = kernel/np.sum(kernel)
    My_Img_2D = conv2D(Orignal_img,kernel_2D)
    CV2_Img_2D = cv2.filter2D(Orignal_img,-1,kernel_2D,borderType=cv2.BORDER_REPLICATE)
    print(f'MSE Between: My_Img_2D, CV2_Img_2D  = {mse(My_Img_2D,CV2_Img_2D)}')
    titles = ['My_Img_2D','CV2_Img_2D']
    images = [My_Img_2D, CV2_Img_2D]
    for i in range(2):
        plt.subplot(1,2,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()
    #------------------------------------------------------------------------------------------------
    print('---------------------------- 2 Image derivatives & blurring ----------------------------:')
    print('# 2.1 Derivatives')
    der_x,der_y,magnitude,directions = convDerivative(Orignal_img)
    titles = ['der_x','der_y','magnitude','directions']
    images = [der_x, der_y, magnitude, directions]

    for i in range(4):
        plt.subplot(1,4,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()

    print('---------------------------- 2.2 Blurring: Bonus ----------------------------:')
    CV2_blur_Img = blurImage2(Orignal_img,5)
    My_blur_Img = blurImage1(Orignal_img,5)
    titles = ['CV2_blur_Img','My_blur_Img','Original_Img']
    images = [CV2_blur_Img,My_blur_Img,Orignal_img]
    for i in range(3):
        plt.subplot(1,3,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()

    print('---------------------------- 3 Edge detection ----------------------------:')
    Sobel_Thresh = 0.4
    Canny_TH1 = Orignal_img.max() * 0.15
    Canny_TL2 = Canny_TH1 * 0.03
    My_Sobel_Img,CV2_Sobel_Img = edgeDetectionSobel(Orignal_img,Sobel_Thresh)
    ZeroCrossingSimple_img = edgeDetectionZeroCrossingSimple(Orignal_img)
    ZeroCrossingLOG_Img = edgeDetectionZeroCrossingLOG(Orignal_img)
    
    My_Canny_Img,Cv2_Canny_Img = edgeDetectionCanny(Orignal_img,Canny_TH1,Canny_TL2)
    print(f'MSE Between: ZeroCrossingSimple_img, ZeroCrossingLOG_Img  = {mse(ZeroCrossingSimple_img,ZeroCrossingLOG_Img)}')
    print(f'MSE Between: My_Sobel_Img, CV2_Sobel_Img  = {mse(My_Sobel_Img,CV2_Sobel_Img)}')
    
    titles = ['My_Sobel_Img','CV2_Sobel_Img','ZeroCrossingSimple_img','ZeroCrossingLOG','My_Canny_Img','Cv2_Canny_Img']
    images = [My_Sobel_Img, CV2_Sobel_Img,ZeroCrossingSimple_img,ZeroCrossingLOG_Img,My_Canny_Img,Cv2_Canny_Img]
    for i in range(6):
        plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()
    img_path = 'coins.jpg'
    coins = cv2.imread(img_path,0)
    hough = houghCircle(coins,60,80)
    print(hough)

if __name__ == '__main__':
    main()