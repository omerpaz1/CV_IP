import numpy as np
import matplotlib.pyplot as plt
import cv2


LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> np.int:
    """
    Return my ID
    :return: int
    """
    return 311326490


def  conv1D(inSignal:np.ndarray,kernel1:np.ndarray)->np.ndarray: 
    """ 
    Convolve  a  1-D  array  with  a  given  kernel 
    :param  inSignal:  1-D  array
    :param  kernel1:  1-D  array  as  a  kernel
    :return: The convolved array 
    """
    kernel1 = kernel1[::-1]
    if(kernel1.shape[0] % 2 == 0):
        pad_size = int((kernel1.shape[0])/2)
        padded_signal = np.pad(inSignal, pad_size, 'constant', constant_values=0)
        new_signal = np.array([])
        # Apply kernel to each pixel
        for i in range(0, padded_signal.shape[0]-pad_size-1):
            sub_signal = padded_signal[i:i+pad_size+1]
            prod = sub_signal*kernel1
            sub_signal_sum = np.sum(prod)
            new_signal = np.append(new_signal, sub_signal_sum)
    else:
        # Padding array with zeros so resulting array is same length as original
        for i in range(0,len(kernel1)):
            signal = inSignal[1:i+1]
            k = kernel1[len(kernel1)-1-i]
            print(f'signal = {signal}')
            print(f'k = {k}')

        pad_size = int((kernel1.shape[0]-1)/2)     
        padded_signal = inSignal
        new_signal = np.array([])
        # Apply kernel to each pixel
        for i in range(pad_size, padded_signal.shape[0]-pad_size):
            sub_signal = padded_signal[i-pad_size:i+pad_size+1]
            prod = (sub_signal*kernel1).sum()
            new_signal = np.append(new_signal, prod)
            # print(f'sub_signal = {sub_signal}')
            # print(f'kernel = {kernel1}')
            # print(f'new_signal = {new_signal}')
    return new_signal



def conv2D(inImage:np.ndarray,kernel2:np.ndarray)->np.ndarray: 
    """
    Convolve  a  2-D  array  with  a  given  kernel
    :param  inImage:  2D  image
    :param  kernel2:  A  kernel
    :return:  The  convolved  image 
    """
    pad_size = int((kernel2.shape[0]-1)/2)
#     padded_image = np.pad(inImage, ((pad_size,pad_size),(pad_size,pad_size)), 'constant', constant_values=0)
    padded_image = np.pad(inImage, ((pad_size,pad_size),(pad_size,pad_size)), 'reflect')
    new_image = np.zeros(padded_image.shape)
    for i in range(pad_size, padded_image.shape[0]-pad_size):
        for j in range(pad_size, padded_image.shape[1]-pad_size):
            sub_image = padded_image[i-pad_size:i+pad_size+1,j-pad_size:j+pad_size+1]
            prod = sub_image*kernel2
            sub_image_sum = np.sum(prod)
            new_image[i,j] = sub_image_sum
    return new_image[pad_size:-pad_size,pad_size:-pad_size]



def  convDerivative(inImage:np.ndarray)  ->  (np.ndarray,np.ndarray,np.ndarray,np.ndarray): 
    """
    Calculate  gradient  of  an  image
    :param  inImage:  Grayscale  iamge
    :return:  (directions,  magnitude,x_der,y_der) 
    """

    k_x = np.array([1,0,-1])
    k_y = np.reshape(np.array([1,0,-1]),(3,1))
    der_X = conv2D(inImage,k_x)
    der_y = conv2D(inImage,k_y)

    directions = np.arctan2(der_X,der_y)
    magnitude = np.sqrt(der_X**2 + der_y**2)
    return (directions,magnitude,der_X,der_y)
    # f, ax = plt.subplots(1,2)
    # ax[0].imshow(gradient,cmap='gray')
    # ax[1].imshow(mag,cmap='gray')
    # plt.show()

def  edgeDetectionSobel(img:  np.ndarray,  thresh:  float  =  0.7)-> (np.ndarray, np.ndarray):
 
    """
    Detects  edges  using  the  Sobel  method
    :param  img:  Input  image
    :param  thresh:  The  minimum  threshold  for  the  edge  response
    :return:  opencv  solution,  my  implementation

    """
    h,w = img.shape[:2]
    ker_Sobol_x = np.array([[1,0,-1],[-2,0,2],[1,0,-1]])
    ker_Sobol_T_x = ker_Sobol_x.T
    ker_Sobol_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    ker_Sobol_T_y = ker_Sobol_y.T
    Smoth_x = conv2D(img,ker_Sobol_x)
    der_x = conv2D(Smoth_x,ker_Sobol_T_x)
    Smoth_y = conv2D(img, ker_Sobol_y)
    der_y = conv2D(Smoth_y, ker_Sobol_T_y)
    magnitude = np.sqrt(der_x**2 + der_y**2)
    magnitude = cv2.normalize(magnitude,magnitude,0.0,1.0,cv2.NORM_MINMAX, cv2.CV_32FC1)
    # for i in range(h):
    #     for j in range(w):
    #         if magnitude[i,j] >= thresh:
    #             magnitude[i,j] = 1
    #         else:
    #             magnitude[i,j] = 0
    # sobol algorigtem:
    sobel_der_x = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    sobel_der_y = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
    magnitude_sobol = np.sqrt(sobel_der_x**2+sobel_der_y**2)
    magnitude_sobol = cv2.normalize(magnitude_sobol,magnitude_sobol,0.0,1.0,cv2.NORM_MINMAX, cv2.CV_32FC1)
    return (magnitude,magnitude_sobol)



def  edgeDetectionZeroCrossingSimple(img:np.ndarray)->(np.ndarray):
    """
    Detecting  edges  using  the  "ZeroCrossingLOG"  method
    :param  I:  Input  image
    :return:  :return:  Edge  matrix """
    ker_gauss = np.array([1,2,1])
    ker_lap = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    h,w = img.shape[:2]
    LoG = cv2.GaussianBlur(img,(5,5),0)
    LoG = conv2D(LoG, ker_lap)
    ZeroCrossingImg = np.zeros(LoG.shape)
    ans2 = np.zeros(LoG.shape)
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            patch = LoG[y-1:y+2, x-1:x+2]
            p = LoG[y, x]
            maxP = patch.max()
            minP = patch.min()
            if (p > 0):
                zeroCross = True if minP < 0 else False
            else:
                zeroCross = True if maxP > 0 else False
            if zeroCross:
                ZeroCrossingImg[y, x] = 1

    return ZeroCrossingImg



def  edgeDetectionCanny(img:  np.ndarray,  thrs_1:  float,  thrs_2:  float)-> (np.ndarray, np.ndarray): 
    """
    Detecting  edges  usint  "Canny  Edge"  method
    :param  img:  Input  image
    :param  thrs_1:  T1
    :param  thrs_2:  T2
    :return:  opencv  solution,  my  implementation 
    """
    h,w = img.shape[:2]
    ker_Sobol_x = np.array([[1,0,-1],[-2,0,2],[1,0,-1]])
    ker_Sobol_T_x = ker_Sobol_x.T
    ker_Sobol_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    ker_Sobol_T_y = ker_Sobol_y.T
    Smoth_x = conv2D(img,ker_Sobol_x)
    der_x = conv2D(Smoth_x,ker_Sobol_T_x)
    Smoth_y = conv2D(img, ker_Sobol_y)
    der_y = conv2D(Smoth_y, ker_Sobol_T_y)
    magnitude = np.sqrt(der_x**2 + der_y**2)
    directions = np.arctan2(der_x,der_y)
    # for i in range(h):
    #     for j in range(w):
    #         if magnitude[i,j] >= thresh:
    #             magnitude[i,j] = 1
    #         else:
    #             magnitude[i,j] = 0