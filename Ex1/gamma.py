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
from ex1_utils import LOAD_GRAY_SCALE
import cv2
import numpy as np
import matplotlib.pyplot as plt

def nothing(x):
    pass

# gamma correction with trackbar.
def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    img = cv2.imread(img_path, rep)
    gamma = 1
    bar_text = "Gamma"
    # Prepare track bar for setting gamma value (prepare with 100 times value because decimal point can not be handled)
    cv2.namedWindow("gammma correction", cv2.WINDOW_NORMAL)
    cv2.createTrackbar(bar_text, "gammma correction", 0, 200, nothing)
    cv2.setTrackbarPos(bar_text, "gammma correction", 100)

    while(1):
        plt.imshow(img)
        # Gamma value acquisition (0 is forcibly pulled back to 0.01)
        gamma = cv2.getTrackbarPos(bar_text, "gammma correction") * 0.01
        if gamma == 0:
            gamma = 0.01
            cv2.setTrackbarPos(bar_text, "gammma correction", 0)

        # Gamma correction lookup table
        look_up_table = np.zeros((256, 1), dtype = 'uint8')
        for i in range(len(look_up_table)):
            look_up_table[i][0] = (len(look_up_table)-1) * pow(float(i) / (len(look_up_table)-1), gamma)

        gamma_correction_image = cv2.LUT(img, look_up_table)

        # Window display
        cv2.putText(gamma_correction_image, "Gamma:" + str(gamma), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0),2)
        cv2.imshow("gammma correction", gamma_correction_image)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cv2.destroyAllWindows()


def main():
    gammaDisplay('beach.jpg', LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()
