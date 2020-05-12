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
    if((kernel2.shape[0] % 2) == 0):
        print('here')
        pad_size  = int(kernel2.shape[0]/2)
        print(pad_size)
    else:
        pad_size = int((kernel2.shape[0]-1)/2)

    padded_image = np.pad(inImage, ((pad_size,pad_size),(pad_size,pad_size)), 'edge')
    new_image = np.zeros(padded_image.shape)
    for i in range(pad_size, padded_image.shape[0]-pad_size):
        #range(2,802)
        for j in range(pad_size, padded_image.shape[1]-pad_size):
            #range(2,1282)
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
    # np.array([1,0,-1])
    k_y = np.reshape(k_x,(3,1))
    der_X = conv2D(inImage,k_x)
    der_y = conv2D(inImage,k_y)

    directions = np.arctan2(der_X,der_y)
    magnitude = np.sqrt(der_X**2 + der_y**2)
    return (directions,magnitude,der_X,der_y)


def  edgeDetectionSobel(img:  np.ndarray,  thresh:  float  =  0.7)-> (np.ndarray, np.ndarray):
 
    """
    Detects  edges  using  the  Sobel  method
    :param  img:  Input  image
    :param  thresh:  The  minimum  threshold  for  the  edge  response
    :return:  opencv  solution,  my  implementation

    """
    ker_Sobol_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    ker_Sobol_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    der_x = conv2D(img,ker_Sobol_x)
    der_y = conv2D(img,ker_Sobol_y)
    x_n_y = np.add(der_x,der_y)
    x_n_y = x_n_y.astype(np.float) / 255
    x_n_y = np.absolute(x_n_y)
    x_n_y[x_n_y > thresh ] = 1
    x_n_y[x_n_y < thresh ] = 0
    # sobol algorigtem:
    ori_sobel_x = cv2.Sobel(img,-1,0,1)
    ori_sobel_y = cv2.Sobel(img,-1,1,0)
    sobol_add =np.add(ori_sobel_x,ori_sobel_y) 
    print(x_n_y)

    return (x_n_y,sobol_add)



def  edgeDetectionZeroCrossingSimple(img:np.ndarray)->(np.ndarray):
    """
    Detecting  edges  using  the  "ZeroCrossingLOG"  method
    :param  I:  Input  image
    :return:  :return:  Edge  matrix """
    ker_lap = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    h,w = img.shape[:2]
    LoG = conv2D(img, ker_lap)
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
    ker_size = 5
    h,w = img.shape[:2]
    ker_Sobol_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    ker_Sobol_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])

    der_x = conv2D(img,ker_Sobol_x)
    der_y = conv2D(img, ker_Sobol_y)

    magnitude = np.sqrt(der_x**2 + der_y**2)
    directions = np.degrees(np.arctan2(der_x,der_y))

    for x in range(h):
        for y in range(w):
            point = directions[x,y] 
            if point < 0:
                directions[x,y] = point+180
            pivot = directions[x,y]
            print(pivot)
            neighbors = directions[y-ker_size:y+ker_size+1,x-ker_size:x+ker_size+1]
    return directions

