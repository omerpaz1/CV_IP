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
    pad_size = len(kernel1)-1
    
    padded_signal = np.pad(inSignal, (pad_size, pad_size), 'constant', constant_values=0)
    print(padded_signal)
    new_signal = np.array([]).astype('int')
    # Apply kernel to each pixel
    for i in range(len(inSignal)+len(kernel1)-1):
          sub_signal = padded_signal[i:pad_size+1+i]
          prod = (sub_signal*kernel1).sum()
          new_signal = np.append(new_signal, prod)
    return new_signal



def conv2D(inImage:np.ndarray,kernel2:np.ndarray)->np.ndarray: 
    """
    Convolve  a  2-D  array  with  a  given  kernel
    :param  inImage:  2D  image
    :param  kernel2:  A  kernel
    :return:  The  convolved  image 
    """
    kernel2 = np.flip(kernel2)
    padded_image = np.pad(inImage, (kernel2.shape[0] // 2, kernel2.shape[1] // 2), 'edge')
    new_image = np.ndarray(inImage.shape)
    for i in range(0, new_image.shape[0]):
        for j in range(0, new_image.shape[1]):
            new_image[i, j] = np.multiply(padded_image[i:i + kernel2.shape[0], j:j + kernel2.shape[1]], kernel2).sum()
    return new_image


def  convDerivative(inImage:np.ndarray)  ->  (np.ndarray,np.ndarray,np.ndarray,np.ndarray): 
    """
    Calculate  gradient  of  an  image
    :param  inImage:  Grayscale  iamge
    :return:  (directions,  magnitude,x_der,y_der) 
    """

    k_x = np.array([1,0,-1])
    k_y = np.reshape(k_x,(3,1))
    der_x = conv2D(inImage,k_x)
    der_y = conv2D(inImage,k_y)

    directions = np.arctan2(der_y,der_x)
    magnitude = np.sqrt(der_x**2 + der_y**2)
    return (directions,magnitude,der_x,der_y)

def blurImage1(in_image: np.ndarray, kernel_size: np.ndarray) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param inImage: Input image
    :param kernelSize: Kernel size
    :return: The Blurred image
    """
def  blurImage2(in_image:np.ndarray,kernel_size:np.ndarray)->np.ndarray: 
    """
    Blur  an  image  using  a  Gaussian  kernel  using  OpenCV  built-in  functions
    :param  inImage:  Input  image
    :param  kernelSize:  Kernel  size
    :return:  The  Blurred  image 
    """
    kernel_x = cv2.getGaussianKernel(kernel_size,0)
    kernel_y = cv2.getGaussianKernel(kernel_size,0)
    k = np.dot(kernel_x,kernel_y.T)
    return cv2.filter2D(in_image,-1,k)

def  edgeDetectionSobel(img:  np.ndarray,  thresh:  float  =  0.7)-> (np.ndarray, np.ndarray):
    """
    Detects  edges  using  the  Sobel  method
    :param  img:  Input  image
    :param  thresh:  The  minimum  threshold  for  the  edge  response
    :return:  opencv  solution,  my  implementation
    """
    ker_Sobol_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    ker_Sobol_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])

    magnitude = np.sqrt(conv2D(img,ker_Sobol_x)**2 + conv2D(img,ker_Sobol_y)**2)
    magnitude = magnitude.astype(np.float) / 255
    magnitude[magnitude > thresh] = 1
    magnitude[magnitude < thresh] = 0

    # sobol algorigtem:
    magnitude_sobel = cv2.magnitude(cv2.Sobel(img,cv2.CV_64F,0,1,thresh),cv2.Sobel(img,cv2.CV_64F,1,0,thresh))
    magnitude_sobel = magnitude_sobel / 255
    magnitude_sobel[magnitude_sobel > thresh] = 1
    magnitude_sobel[magnitude_sobel < thresh] = 0

    return magnitude,magnitude_sobel


def  edgeDetectionZeroCrossingLOG(img:np.ndarray)->(np.ndarray):
    """
    Detecting  edges  using  the  "ZeroCrossingLOG"  method
    :param  I:  Input  image
    :return:  :return:  Edge  matrix 
    """
    return edgeDetectionZeroCrossingSimple(cv2.GaussianBlur(img,(5,5),0))


def  edgeDetectionZeroCrossingSimple(img:np.ndarray)->(np.ndarray):
    """
    Detecting  edges  using  the  "ZeroCrossingLOG"  method
    :param  I:  Input  image
    :return:  :return:  Edge  matrix """
    imga = cv2.GaussianBlur(img,(5,5),0)
    ker_lap = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    LOG = conv2D(imga, ker_lap)
    minLoG = cv2.morphologyEx(LOG, cv2.MORPH_ERODE, np.ones((3,3)))
    maxLoG = cv2.morphologyEx(LOG, cv2.MORPH_DILATE, np.ones((3,3)))
    zeroCross = np.logical_or(np.logical_and(minLoG < 0,  LOG > 0), np.logical_and(maxLoG > 0, LOG < 0)).astype('int')
    return zeroCross



def  edgeDetectionCanny(img:  np.ndarray,  thrs_1:  float,  thrs_2:  float)-> (np.ndarray, np.ndarray): 
    """
    Detecting  edges  usint  "Canny  Edge"  method
    :param  img:  Input  image
    :param  thrs_1:  T1
    :param  thrs_2:  T2
    :return:  opencv  solution,  my  implementation 
    """
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    
    Ix = conv2D(img, Kx)
    Iy = conv2D(img, Ky)
    
    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)
    img_non_max = non_max_suppression(G,theta)
    res, weak, strong = threshold(img_non_max,thrs_1,thrs_2)
    ans = hysteresis(res, weak, strong)
    return ans


def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M,N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    
    for i in range(1,M-1):
        for j in range(1,N-1):
            try:
                q = 255
                r = 255
                
               #angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                #angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                #angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                #angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]

                if (img[i,j] >= q) and (img[i,j] >= r):
                    Z[i,j] = img[i,j]
                else:
                    Z[i,j] = 0

            except IndexError as e:
                pass
    
    return Z

def threshold(img, highThreshold, lowThreshold):

    M, N = img.shape
    res = np.zeros((M,N), dtype=np.int32)
    
    weak = 25
    strong = 255
    
    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)
    
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
    
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    return (res, weak, strong)



def hysteresis(img, weak, strong):
    M, N = img.shape  
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img


def  houghCircle(img:np.ndarray,min_radius:float,max_radius:float)->list:
    """
    Find  Circles  in  an  image  using  a  Hough  Transform  algorithm  extension
    :param  I:  Input  image
    :param  minRadius:  Minimum  circle  radius
    :param  maxRadius:  Maximum  circle  radius
    :return: A list containing the detected circles, [(x,y,radius),(x,y,radius),...]
    """
