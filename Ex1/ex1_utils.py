"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import List

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2

def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 311326490


def imReadAndConvert(filename:str, representation:int)->np.ndarray:
    if representation == LOAD_GRAY_SCALE:
        img = cv2.imread(filename,0)
        norm_img = normalize(img)
    elif representation == LOAD_RGB:
        img = cv2.imread(filename)
        img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        norm_img = normalize(img_rgb)
    return norm_img

# normalize the image between [0,1] in a float type.
def normalize(img):
    norm_img = img/255
    return norm_img

# unnormalize the image between [0,255] in a int32 type.
def unnormalize(img):
    norm_img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return norm_img.astype('uint8')


# disply the image, using plot.
def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    img = imReadAndConvert(filename,representation)
    plt.imshow(img,cmap='gray')
    plt.show()

#tranfrom the image from RGB to YIQ using multiply matrix.
def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    yiq_mat = np.array([[0.299, 0.587, 0.144], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])
    imgYIQ = np.dot(imgRGB, yiq_mat.T.copy())
    return imgYIQ

#tranfrom the image from YIQ to RGB using multiply matrix.
def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    rgb_mat = np.array([[1, 0.956, 0.619], [1, -0.272, -0.647], [1, -1.106, 1.703]])
    imgRGB = np.dot(imgYIQ, rgb_mat.T.copy())
    return imgRGB

# histogram for image grayscle and RGB. 
    # histogram for GRAYSCALE
    # imgEq - > the final image after eq.
    # histOrg - > the histogram of the original image.
    # histEq - > the histogram after eq.
""" 
 histogram for image grayscle and RGB. 
 histogram for GRAYSCALE
 imgEq - > the final image after eq.
 histOrg - > the histogram of the original image.
 histEq - > the histogram after eq.
 """
def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    if len(imgOrig.shape) == 2:
        un_norm_img = unnormalize(imgOrig)
        histOrg,bins = np.histogram(un_norm_img.flatten(),256,[0,256])
        cdf = histOrg.cumsum()
        cdf_normalized = cdf * histOrg.max()/ cdf.max()
        cdf_m = np.ma.masked_equal(cdf,0)
        cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
        cdf = np.ma.filled(cdf_m,0).astype('uint8')
        imgEq = cdf[un_norm_img]
        histEq,bins2 = np.histogram(imgEq.flatten(),256,[0,256])
        imgEq = normalize(imgEq)
        return (imgEq, histOrg, histEq)
    else:
        imgYIQ = transformRGB2YIQ(imgOrig)
        un_norm_img = unnormalize(imgYIQ[:,:,0])
        y_min = np.amin(un_norm_img)
        y_max = np.amax(un_norm_img)
        histOrg,bins = np.histogram(un_norm_img.flatten(),256,[0,256])
        cdf = histOrg.cumsum()
        cdf_normalized = cdf * histOrg.max()/ cdf.max()

        cdf_m = np.ma.masked_equal(cdf,0)
        cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
        cdf = np.ma.filled(cdf_m,0).astype('uint8')
        res = cdf[un_norm_img.astype('uint8')]
        histEq,bins2 = np.histogram(res.flatten(),256,[0,256])
        res = normalize(res)
        imgYIQ[:,:,0] = res
        imgEq = transformYIQ2RGB(imgYIQ)
        return (imgEq, histOrg, histEq)

# helping function the plot image.
def plt_hist(hist, img):
    un_norm_img = unnormalize(img)
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()
    plt.plot(cdf_normalized, color = 'b')
    plt.hist(un_norm_img.flatten(),255,[0,255], color = 'r')
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    plt.show()




def quantizeImage(imgOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imgOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    RGBimg = False
    if len(imgOrig.shape) == 3:
        RGBimg = True
        imgYIQ = transformRGB2YIQ(imgOrig)
        imgOrig = imgYIQ[:,:,0]

    un_norm_img = unnormalize(imgOrig)
    histImage,bins = np.histogram(un_norm_img.flatten(),256,[0,256])
    
    qunt_img = un_norm_img.copy()
    qImage_i = []
    error_i = []
    q = [i for i in range(nQuant)]
    error_i = []

    #  all the iterations
    for i in range(nIter):
        # print(f'----- iter = {i} -----')
        if(i == 0): # if its the first round we need to split the budries.
            z = np.arange(0, 255 - int(256 / nQuant) + 1, int(256 / nQuant))
            z = np.append(z, 255)
            Intensities = np.array(range(256))
        else: # if this is not the firsr round
            for i in range(1,len(z)-2): # nQuant = 3 
                temp = int((q[i-1]+q[i])/2)
                if temp != z[i-1] and temp != z[i+1]:
                    z[i] = temp

        for j in range(len(z)-1): # calculate the q between.
            val = np.logical_and((z[j] < qunt_img), (qunt_img < z[j+1]))  # the current cluster
            if j is not (len(z)-2):
                q[j] = np.average(Intensities[z[j]:z[j+1]],weights=histImage[z[j]:z[j+1]])
                qunt_img[val] = q[j] 
            else:
                q[j] = np.average(Intensities[z[j]:z[j+1]+1],weights=histImage[z[j]:z[j+1]+1])
                qunt_img[val] = q[j] 
        # calculate rmse
        rmse = np.mean(np.power(qunt_img - un_norm_img,2))
        error_i.append(rmse)
        # print(f"rmse = {rmse}")

        if RGBimg:
           temp = qunt_img/255
           imgYIQ[:,:,0] = temp
           temp = transformYIQ2RGB(imgYIQ)
           qImage_i.append(temp)
        else:
            temp = qunt_img/255
            qImage_i.append(temp)

    return qImage_i ,error_i
    

