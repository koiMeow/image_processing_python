import numpy as np
import cv2

def main():
    img = cv2.imread('padded_gray_img.jpg')

    gauss_blur = cv2.GaussianBlur(img, (7, 7), 0)
    cv2.imwrite('gauss_blurred.jpg', gauss_blur)

if __name__ == '__main__':
    main()