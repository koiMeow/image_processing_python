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
    gray_scaled_img = gray_scaling(input_img)
    height, width, channels = gray_scaled_img.shape
    result = np.zeros((height, width, 1))
    mask_size = len(mask)
    padding = mask_size//2 + 1

    for i in range(height-padding):
        for j in range(width-padding):
            current_pixel_value = 0
            for x in range(mask_size):
                for y in range(mask_size):
                    current_pixel_value += gray_scaled_img[i+x, j+y] * mask[x][y]
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

def main():
    edge_detector = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]

    img = cv2.imread('gura.jpg')

    # conv_img = convolution(img, edge_detector)

    # cv2.imwrite('conv_img.png', conv_img)

    max_pooling_img = max_pooling(img, 2, 2)
    cv2.imwrite('max_pooling_img.png', max_pooling_img)

if __name__ == "__main__":
    main()