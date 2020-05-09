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
    x_der = conv2D(inImage,k_x)
    y_der = conv2D(inImage,k_y)

    directions = np.arctan2(x_der,y_der)
    magnitude = np.sqrt(x_der**2 + y_der**2)
    return (directions,magnitude,x_der,y_der)
    # f, ax = plt.subplots(1,2)
    # ax[0].imshow(gradient,cmap='gray')
    # ax[1].imshow(mag,cmap='gray')
    # plt.show()
    # f, ax = plt.subplots(1,2)
    # ax[0].imshow(x_der,cmap='gray')
    # ax[1].imshow(y_der,cmap='gray')
    # plt.show()