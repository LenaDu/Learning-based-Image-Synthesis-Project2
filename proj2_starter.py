# --------------------------------------------------------
# Written by Yufei Ye and modified by Sheng-Yu Wang (https://github.com/JudyYe)
# Convert from MATLAB code https://inst.eecs.berkeley.edu/~cs194-26/fa18/hw/proj3/gradient_starter.zip
# --------------------------------------------------------
from __future__ import print_function

import argparse
import numpy as np
import cv2
import imageio
import matplotlib.pyplot as plt
from scipy.sparse import linalg, lil_matrix
import time

def toy_recon(image):
    imh, imw = image.shape
    im2var = np.arange(imh * imw).reshape((imh, imw)).astype(int)
    A = lil_matrix((2 * (imw) * (imh), (imw) * (imh)))
    b = np.zeros(2 * (imw) * (imh))

    offset = (imw - 1) * (imh - 1)

    # x gradients & y gradients
    e = 0
    for y in range(imh - 1):
        for x in range(imw - 1):
            A[e, im2var[y, x + 1]] = 1
            A[e, im2var[y, x]] = -1

            A[e + offset, im2var[y + 1, x]] = 1
            A[e + offset, im2var[y, x]] = -1

            b[e] = image[y, x + 1] - image[y, x]
            b[e + offset] = image[y + 1, x] - image[y, x]

            e += 1

    A[-1, im2var[0, 0]] = 1
    b[-1] = image[0, 0]
    v = linalg.lsqr(A, b)
    return v[0].reshape((imh, imw))

def poisson_blend(fg, mask, bg):
    """
    Poisson Blending.
    :param fg: (H, W, C) source texture / foreground object
    :param mask: (H, W, 1)
    :param bg: (H, W, C) target image / background
    :return: (H, W, C)
    """
    mask_area = np.flatnonzero(mask)
    nonzero_num = len(mask_area)
    imh, imw, channel = bg.shape
    im2var = np.arange(imh * imw).reshape((imh, imw)).astype(int)
    result = np.zeros((nonzero_num, channel))

    for c in range(channel):
      bg_img = bg[:,:,c]
      src_img = fg[:,:,c]
      A = lil_matrix((4 * nonzero_num + 1, nonzero_num))
      b = np.zeros(4 * nonzero_num + 1)
      e = 0
      for row in range(imh-1):
        for col in range(imw-1):
          if im2var[row, col] not in mask_area:
            continue
          directions = [(row, col-1), (row, col+1), (row-1, col), (row+1, col)]
          for direction in directions:
            A[e, np.where(mask_area == im2var[row,col])[0][0]] = 1
            if im2var[direction] in mask_area:
              A[e, np.where(mask_area == im2var[direction])[0][0]] = -1
              b[e] = src_img[row, col] - src_img[direction]
            else:
              b[e] = bg_img[direction] + src_img[row, col] - src_img[direction]
            e += 1

      v = linalg.lsqr(A,b)
      v = np.clip(v[0], 0, 1)
      result[:,c] = v

    output = np.zeros((imh, imw, channel))
    for row in range(imh):
      for col in range(imw):
        if im2var[row, col] in mask_area:
          for c in range(channel):
            output[row, col, c] = result[np.where(mask_area == im2var[row,col])[0][0], c]
        else:
          for c in range(channel):
            output[row, col, c] = bg[row, col, c]
    return output

def mixed_blend(fg, mask, bg):
    """EC: Mix gradient of source and target"""
    mask_area = np.flatnonzero(mask)
    nonzero_num = len(mask_area)
    imh, imw, channel = bg.shape
    im2var = np.arange(imh * imw).reshape((imh, imw)).astype(int)
    result = np.zeros((nonzero_num, channel))

    for c in range(channel):
      bg_img = bg[:,:,c]
      src_img = fg[:,:,c]
      A = lil_matrix((4 * nonzero_num + 1, nonzero_num))
      b = np.zeros(4 * nonzero_num + 1)
      e = 0
      for row in range(imh-1):
        for col in range(imw-1):
          if im2var[row, col] not in mask_area:
            continue
          directions = [(row, col-1), (row, col+1), (row-1, col), (row+1, col)]
          for direction in directions:
            A[e, np.where(mask_area == im2var[row,col])[0][0]] = 1
            src_gradient = src_img[row, col] - src_img[direction]
            tgt_gradient = bg_img[row,col] - bg_img[direction]
            # Gradient
            if im2var[direction] in mask_area:
              A[e, np.where(mask_area == im2var[direction])[0][0]] = -1
              if abs(src_gradient) >= abs(tgt_gradient):
                b[e] = src_gradient
              else:
                b[e] = tgt_gradient
            else:
              if abs(src_gradient) >= abs(tgt_gradient):
                b[e] = bg_img[direction] + src_gradient
              else:
                b[e] = bg_img[direction] + tgt_gradient
            e += 1

      v = linalg.lsqr(A,b)
      # v = np.clip(v[0], 0, 255).astype(int)
      v = np.clip(v[0], 0, 1)
      result[:,c] = v

    # output = np.zeros((imh, imw, channel), dtype=int)
    output = np.zeros((imh, imw, channel))
    for row in range(imh):
      for col in range(imw):
        if im2var[row, col] in mask_area:
          for c in range(channel):
            output[row, col, c] = result[np.where(mask_area == im2var[row,col])[0][0], c]
        else:
          for c in range(channel):
            output[row, col, c] = bg[row, col, c]
    return output


def color2gray(rgb_image):
    """Naive conversion from an RGB image to a gray image."""
    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)


def mixed_grad_color2gray(rgb_image):
    """EC: Convert an RGB image to gray image using mixed gradients."""
    return np.zeros_like(rgb_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Poisson blending.")

    start_time = time.time()
    parser.add_argument("-q", "--question", required=True, choices=["toy", "blend", "mixed", "color2gray"])
    args, _ = parser.parse_known_args()

    # Example script: python proj2_starter.py -q toy
    if args.question == "toy":
        image = imageio.imread('./data/toy_problem.png') / 255.
        image_hat = toy_recon(image)

        plt.subplot(121)
        plt.imshow(image, cmap='gray')
        plt.title('Input')
        plt.subplot(122)
        plt.imshow(image_hat, cmap='gray')
        plt.title('Output')
        plt.show()

    # Example script: python proj2_starter.py -q blend -s data/source_01_newsource.png -t data/target_01.jpg -m data/target_01_mask.png
    if args.question == "blend":
        parser.add_argument("-s", "--source", required=True)
        parser.add_argument("-t", "--target", required=True)
        parser.add_argument("-m", "--mask", required=True)
        args = parser.parse_args()

        # after alignment (masking_code.py)
        ratio = 1
        fg = cv2.resize(imageio.imread(args.source), (0, 0), fx=ratio, fy=ratio)
        bg = cv2.resize(imageio.imread(args.target), (0, 0), fx=ratio, fy=ratio)
        mask = cv2.resize(imageio.imread(args.mask), (0, 0), fx=ratio, fy=ratio)

        fg = fg / 255.
        bg = bg / 255.
        mask = (mask.sum(axis=2, keepdims=True) > 0)

        blend_img = poisson_blend(fg, mask, bg)

        plt.subplot(121)
        plt.imshow(fg * mask + bg * (1 - mask))
        plt.title('Naive Blend')
        plt.subplot(122)
        plt.imshow(blend_img)
        plt.title('Poisson Blend')
        plt.show()

    if args.question == "mixed":
        parser.add_argument("-s", "--source", required=True)
        parser.add_argument("-t", "--target", required=True)
        parser.add_argument("-m", "--mask", required=True)
        args = parser.parse_args()

        # after alignment (masking_code.py)
        ratio = 1
        fg = cv2.resize(imageio.imread(args.source), (0, 0), fx=ratio, fy=ratio)
        bg = cv2.resize(imageio.imread(args.target), (0, 0), fx=ratio, fy=ratio)
        mask = cv2.resize(imageio.imread(args.mask), (0, 0), fx=ratio, fy=ratio)

        fg = fg / 255.
        bg = bg / 255.
        mask = (mask.sum(axis=2, keepdims=True) > 0)

        blend_img = mixed_blend(fg, mask, bg)
        print("--- %s seconds ---" % (time.time() - start_time))

        plt.subplot(121)
        plt.imshow(fg * mask + bg * (1 - mask))
        plt.title('Naive Blend')
        plt.subplot(122)
        plt.imshow(blend_img)
        plt.title('Mixed Blend')
        plt.show()

    if args.question == "color2gray":
        parser.add_argument("-s", "--source", required=True)
        args = parser.parse_args()

        rgb_image = imageio.imread(args.source)
        gray_image = color2gray(rgb_image)
        mixed_grad_img = mixed_grad_color2gray(rgb_image)

        plt.subplot(121)
        plt.imshow(gray_image, cmap='gray')
        plt.title('rgb2gray')
        plt.subplot(122)
        plt.imshow(mixed_grad_img, cmap='gray')
        plt.title('mixed gradient')
        plt.show()

    plt.close()
