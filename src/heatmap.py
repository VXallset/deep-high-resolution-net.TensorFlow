"""
This file is used to generate the heat map and other stuffs.

@ Author: Yu Sun. vxallset@outlook.com

@ Date created: Jun 04, 2019

@ Last modified: Jun 27, 2019

"""
import numpy as np
from skimage import io, draw
from dataset import draw_lines_on_img


def gaussian_kernel(kernel_length=3, sigma=1.):
    """
    creates gaussian kernel with side length l and a sigma of sig
    """

    ax = np.arange(-kernel_length // 2 + 1., kernel_length // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))

    return kernel / np.sum(kernel)


def calculate_groundtruth_heatmap(keypoint_ver, keypoint_hor, kepoint_class, kernel_length=3, sigma=1.0):
    batch_size, keypoints_number = kepoint_class.shape
    assert kernel_length % 2 == 1, 'kernel_length must be odd!'
    kernel = gaussian_kernel(kernel_length=kernel_length, sigma=sigma)
    half_length = kernel_length // 2
    heatmap = np.zeros((batch_size, 256, 192, keypoints_number), dtype=np.float32)

    for b in range(batch_size):
        for n in range(keypoints_number):
            # if the keypoint class is 3, continue
            if kepoint_class[b, n] == 3:
                continue

            for i in range(-half_length, half_length + 1):
                for j in range(-half_length, half_length + 1):
                    if keypoint_ver[b, n] + i >= 256 or keypoint_ver[b, n] + i < 0 \
                            or keypoint_hor[b, n] + j >= 192 or keypoint_hor[b, n] + j < 0:
                        continue
                    heatmap[b, keypoint_ver[b, n] + i, keypoint_hor[b, n] + j, n] += kernel[i + half_length, j + half_length]
    return heatmap


def decode_output(net_output, threshold=0.0):
    batch_size, size_ver, size_hor, keypoints_number = net_output.shape
    kp_ver = np.zeros((batch_size, keypoints_number))
    kp_hor = np.zeros_like(kp_ver)
    kp_class = np.ones_like(kp_hor) * 3

    for b in range(batch_size):
        for n in range(keypoints_number):
            max_index = np.argmax(net_output[b, :, :, n])
            max_row = max_index // 192
            max_col = max_index % 192
            if net_output[b, max_row, max_col, n] > threshold:
                # print(net_output[b, max_row, max_col, n])
                kp_ver[b, n] = max_row
                kp_hor[b, n] = max_col
                kp_class[b, n] = 1
    prediction = np.zeros((batch_size, keypoints_number * 3))
    prediction[:, ::3] = kp_ver
    prediction[:, 1::3] = kp_hor
    prediction[:, 2::3] = kp_class
    return prediction


def decode_pose(images, net_output, threshold=0.001):
    # key point class: 1:visible, 2: invisible, 3: not marked
    prediction = decode_output(net_output, threshold=threshold)

    batch_size, size_ver, size_hor, keypoints_number = net_output.shape
    kp_ver = prediction[:, ::3]
    kp_hor = prediction[:, 1::3]
    kp_class = prediction[:, 2::3]

    for b in range(batch_size):
        point_hor = kp_hor[b]
        point_ver = kp_ver[b]
        point_class = kp_class[b]
        images[b, :, :, :] = draw_lines_on_img(images[b], point_ver, point_hor, point_class)
        for i in range(len(point_class)):
            if point_class[i] != 3:
                rr, cc = draw.circle(point_ver[i], point_hor[i], 10, (256, 192))
                images[b, rr, cc, :] = 0

    return images


def calculate_distance(prediction, groundtruth):
    kp_ver_pred = prediction[:, ::3]
    kp_hor_pred = prediction[:, 1::3]
    kp_class_pred = prediction[:, 2::3]

    kp_ver_gt = groundtruth[:, ::3]
    kp_hor_gt = groundtruth[:, 1::3]
    kp_class_gt = groundtruth[:, 2::3]

    distance2 = (kp_ver_gt - kp_ver_pred) ** 2 + (kp_hor_gt - kp_hor_pred) ** 2
    mask = np.array(kp_class_gt != 3, dtype=np.int)
    result = np.sqrt(distance2) * mask
    return result
