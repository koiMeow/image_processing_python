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
            result[i, j] = input_img[i-1, j-1, 0]
    
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
            result[i-1, j-1] = current_pixel_value

    return result

# Median filter
def median_filter(input_img):
    height, width, channels = input_img.shape
    result = np.zeros((height, width, 1))
    padded = padding_zero(input_img)

    for i in range(1, height+1):
        for j in range(1, width+1):
            current_pixel_value = []
            for x in range(-1, 2):
                for y in range(-1, 2):
                    current_pixel_value.append(padded[i+x, j+y])
            current_pixel_value.sort()
            result[i-1, j-1] = current_pixel_value[4]

    return result

# Draw histogram
def draw_hist(input_img, title):
    plt.hist(input_img.ravel(), 256, [0,255])
    plt.title(title)
    plt.savefig(title)
    plt.close()

def main():
    mean_filter_mask = np.array([[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]])

    n_img = cv2.imread('noise_image.png')
    mean_img = convolution(n_img, mean_filter_mask)
    median_img = median_filter(n_img)

    cv2.imwrite('output1.png', mean_img)
    cv2.imwrite('output2.png', median_img)

    draw_hist(n_img, "noise_image_his.png")
    draw_hist(mean_img, "output1_his.png")
    draw_hist(median_img, "output2_his.png")

if __name__ == '__main__':
    main()