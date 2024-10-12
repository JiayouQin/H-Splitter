import numpy as np
import cv2
import skimage
def get_line_pixels(w, h, x1, y1, x2, y2):
    # Get the coordinates of the pixels along the line
    rr, cc = skimage.draw.line(y1, x1, y2, x2)  # (row, col) -> (y, x)
    # Filter out coordinates that are outside the image bounds
    valid_idx = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
    return np.column_stack((cc[valid_idx], rr[valid_idx]))

def img_uint8(image):
    image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
    image = image.astype(np.uint8)
    return image


def img_resize(image, size):
    height, width = image.shape[:2]
    crop_size = min(height, width)
    start_y = (height - crop_size) // 2
    start_x = (width - crop_size) // 2
    center_cropped_image = image[start_y:start_y+crop_size, start_x:start_x+crop_size]
    center_cropped_image = cv2.resize(center_cropped_image,(size, size))
    return center_cropped_image


def generate_gaussian_image(height, width, center_point, sigma):
    # Create a grid of coordinates
    (center_x, center_y) = center_point
    x = np.arange(width) - center_x
    y = np.arange(height) - center_y
    x, y = np.meshgrid(x, y)

    # Calculate the squared distances from the center
    distances_squared = x ** 2 + y ** 2

    # Calculate the Gaussian kernel
    kernel = np.exp(-distances_squared / (2 * sigma ** 2))

    # Normalize the kernel to the range [0, 1]
    kernel_normalized = (kernel - np.min(kernel)) / (np.max(kernel) - np.min(kernel))

    return kernel_normalized


def SVD_keypoint(keypoint_list_1, keypoint_list_2):
    center_1 = np.mean(keypoint_list_1, axis=0)
    center_2 = np.mean(keypoint_list_2, axis=0)

    new_keypoint_list_1 = keypoint_list_1 - center_1
    new_keypoint_list_2 = keypoint_list_2 - center_2

    M = new_keypoint_list_2.T @ new_keypoint_list_1
    u, s, vt = np.linalg.svd(M)

    R = u @ vt
    if np.linalg.det(R) < 0:
        u[:, -1] *= -1  # Adjust the last column of u
        R = u @ vt

    T = center_2 - R @ center_1

    # Create a homogeneous transformation matrix
    transform_matrix = np.eye(3)  # Start with an identity matrix
    transform_matrix[0:2, 0:2] = R  # Insert R into the top-left
    transform_matrix[0:2, 2] = T   # Insert T into the top-right

    return transform_matrix, [u, s, vt]