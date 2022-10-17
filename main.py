from cv2 import hconcat
import numpy as np
import cv2

# Gray scale transform
def gray_scaling(input_img):
    height, width, channels = input_img.shape
    result = np.zeros((height, width, 1))

    for i in range(height):
        for j in range(width):
            result[i, j] = input_img[i, j, 2] * 0.299 + input_img[i, j, 1] * 0.587 + input_img[i, j, 0] * 0.114
    
    return result

# Convolution
def convolution(input_img, mask):
    height, width, channels = input_img.shape
    result = np.zeros((height, width, channels))
    mask_size = len(mask)
    padding = mask_size//2 + 1

    for i in range(height-padding):
        for j in range(width-padding):
            current_pixel_value = 0
            for x in range(mask_size):
                for y in range(mask_size):
                    current_pixel_value += input_img[i+x, j+y] * mask[x][y]
            result[i, j] = max(0, current_pixel_value)

    return result

# Max pooling for gray level image
def max_pooling(input_img, mask_size, stride):
    height, width, channels = input_img.shape
    result = np.zeros((height, width, channels))

    for i in range(0, height-stride, stride):
        for j in range(0, width-stride, stride):
            max_pixel_values = 0
            for x in range(mask_size):
                for y in range(mask_size):
                    max_pixel_values = max(input_img[i+x, j+y], max_pixel_values)
            result[i//2, j//2] = max_pixel_values
    
    return result

# Binarization for gray level image
def binarization(input_img, threshold):
    height, width, channels = input_img.shape
    result = np.zeros((height, width, channels))

    for i in range(height):
        for j in range(width):
            if input_img[i, j] >= threshold:
                result[i, j] = 255
            else:
                result[i, j] = 0
    
    return result

# Pixel value oppsite
def opposite(input_img):
    height, width, channels = input_img.shape
    result = np.zeros((height, width, channels))

    for i in range(height):
        for j in range(width):
            result[i, j] = (-1) * (0 - input_img[i, j])
    
    return result
                

def main():
    edge_detector = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
    low_pass_filter = [[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]]

    img = cv2.imread('gura.jpg')

    gray_img = gray_scaling(img)
    cv2.imwrite("gray_img.jpg", gray_img)
    conv_img = convolution(gray_img, edge_detector)
    cv2.imwrite("edge_detected.jpg", conv_img)
    bina_img = binarization(conv_img, 40)
    cv2.imwrite("binarization.jpg", bina_img)
    low_passed_img = convolution(bina_img, low_pass_filter)
    cv2.imwrite("low_passed.jpg", low_passed_img)
    second_bina_img = binarization(low_passed_img, 40)
    cv2.imwrite("second_bina.jpg", second_bina_img)

    result = hconcat([gray_img, conv_img, bina_img, low_passed_img, second_bina_img])
    cv2.imwrite("result.jpg", result)

if __name__ == "__main__":
    main()