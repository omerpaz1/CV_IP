
import numpy as np
from ex1_utils import *
nQuant = 4
import matplotlib.pyplot as plt
# for i in range(1,4):

# img[img<z[i] and img>z[i-1]]=q[0]


z = [0 , 149 , 220 , 255]
q = [99, 199, 241]


img = np.random.randint(0,255,(10,20))
norm_img = cv2.normalize(img.astype('double'), None, 0.0, 1.0, cv2.NORM_MINMAX)
norm_img = img/255
print((img))

print("********************")

print((norm_img*255).astype('uint8'))

# print(img)
# copyImg = np.copy(img)
# print("=====================================")
# print(len(z))
# for i in range(1,4):
#     start = z[i-1]
#     end = z[i]
#     val = q[i-1]
#     print(f"start = {z[i-1]} , end = {z[i]} q[i-1] = {q[i-1]}")
#     for k in range(start,end):
#          copyImg[img == k] = val
# rmse = np.mean(np.power(img - copyImg,2))
# print(f"rmse = {rmse}")
