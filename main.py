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

def padding(input_img, padding_size):
    height, width, channels = input_img.shape
    result = np.zeros((height + padding_size*2, width + padding_size*2, channels))
    for i in range(padding_size, height+padding_size*2-1):
        for j in range(padding_size, width+padding_size*2-1):
            result[i, j] = input_img[i-1, j-1]

    return result

# Convolution
def convolution(input_img, mask):
    height, width, channels = input_img.shape
    result = np.zeros((height, width, channels))
    padding_size = len(mask) // 2
    padded_input_img = padding(input_img, padding_size)

    for i in range(padding_size, height+padding_size-2):
        for j in range(padding_size, width+padding_size-2):
            current_pixel_value = 0
            for x in range(-padding_size, padding_size+1):
                for y in range(-padding_size, padding_size+1):
                    current_pixel_value += padded_input_img[i+x, j+y] * mask[x+padding_size][y+padding_size]
            result[i-padding_size, j-padding_size] = max(0, current_pixel_value)

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
            result[i, j] = 255-input_img[i, j]
    
    return result
                
# Sobel gradient calculation
def sobel(v_img, h_img):
    height, width, channels = v_img.shape
    result = np.zeros((height, width, channels))

    for i in range(height):
        for j in range(width):
            result[i, j] = (v_img[i, j]**2 + h_img[i, j]**2) ** 0.5
    
    return result

def main():
    edge_detector = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
    sobel_detector_vertical = [[-1, 0 ,1], [-2, 0, 2], [-1, 0, 1]]
    sobel_detector_horizontal = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    low_pass_filter = [[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]]

    img = cv2.imread('gura.jpg')

    gray_img = gray_scaling(img)
    cv2.imwrite("gray_img.jpg", gray_img)

    vedge_img = convolution(gray_img, sobel_detector_vertical)
    hedge_img = convolution(gray_img, sobel_detector_horizontal)
    sobel_img = sobel(vedge_img, hedge_img)
    bina_img = binarization(sobel_img, 30)

    bina_img_list = []
    for i in range(1, 251, 10):
        bina_img_list.append(opposite(binarization(sobel_img, i)))
    
    bina_result = cv2.hconcat(bina_img_list)
    cv2.imwrite("bina_result.jpg", bina_result)

    result = cv2.hconcat([gray_img, sobel_img, opposite(sobel_img), bina_img, opposite(bina_img)])
    cv2.imwrite("result.jpg", result)

if __name__ == "__main__":
    main()