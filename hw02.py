from enum import auto
import numpy as np
import cv2
import matplotlib.pyplot as plt

# RGB to gray scale
def gray_scaling(input_img):
    height, width, channels = input_img.shape
    result = np.zeros((height, width, 1))

    for i in range(height):
        for j in range(width):
            result[i, j] = input_img[i, j, 2] * 0.299 + input_img[i, j, 1] * 0.587 + input_img[i, j, 0] * 0.114
    
    return result

# Padding zero
def padding_zero(input_img):
    height, width, channels = input_img.shape
    result = np.zeros((height+2, width+2, 1))

    for i in range(1, height+1):
        for j in range(1, width+1):
            result[i, j] = input_img[i-1, j-1]
    
    return result

# Convolution
def convolution(input_img, mask):
    height, width, channels = input_img.shape
    result = np.zeros((height, width, 1))
    padded = padding_zero(input_img)

    for i in range(1, height+1):
        for j in range(1, width+1):
            current_pixel_value = 0
            for x in range(-1, 2):
                for y in range(-1, 2):
                    current_pixel_value += padded[i+x, j+y] * mask[x+1][y+1]
            # result[i-1, j-1] = np.sum(np.multiply(padded[i-1:i+2, j-1:j+2], mask))
            result[i-1, j-1] = current_pixel_value

    return result

# Draw histogram
def draw_hist(input_img):
    height, width, channels = input_img.shape
    pixel_values = np.zeros((256))
    
    for i in range(height):
        for j in range(width):
            pixel_values[input_img[i, j, 0]] += 1
    
    plt.hist(pixel_values, bins='auto')
    plt.show()

def main():
    sobel_detector_y = np.array([[-1, 0 ,1], [-2, 0, 2], [-1, 0, 1]])
    sobel_detector_x = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    img = cv2.imread('gura.jpg')
    noise_img = cv2.imread('noise_image.png')

    draw_hist(noise_img)

if __name__ == '__main__':
    main()