import numpy as np
from keras.preprocessing import image
from scipy.signal import convolve2d

def sobel_operator(image_path, target_size = (128, 128)):
    
    img = image.load_img(image_path, grayscale = True, target_size = target_size)
    img_arr = np.reshape(image.img_to_array(img), (target_size[0], target_size[1]))
    
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    c_img_x = convolve2d(img_arr, sobel_x)
    c_img_y = convolve2d(img_arr, sobel_y)
    
    img_size = np.shape(c_img_x)
    c_img = np.reshape(np.sqrt(c_img_x**2 + c_img_y**2), (img_size[0], img_size[1], 1))
        
    return image.array_to_img(c_img)

# Ie. -- c_img = sobel_operator('my_image.jpg', target_size = (256, 256))
