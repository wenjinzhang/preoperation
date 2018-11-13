#!/usr/bin/python
# coding:utf8

import numpy as np
import cv2

def shi_tomasi(image):
    # Converting to grayscale
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Specifying maximum number of corners as 1000
    # 0.01 is the minimum quality level below which the corners are rejected
    # 10 is the minimum euclidean distance between two corners
    corners_img = cv2.goodFeaturesToTrack(gray_img, 1000, 0.01, 10)

    corners_img = np.int0(corners_img)

    for corners in corners_img:
        x, y = corners.ravel()
        # Circling the corners in green
        cv2.circle(image, (x, y), 3, [0, 255, 0], -1)

    return image


if __name__ == '__main__':
    cam = cv2.VideoCapture(0)
    ret, prev = cam.read()
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    while True:
        ret, img = cam.read()
        img = shi_tomasi(img)
        cv2.imshow('flow', img)

        ch = cv2.waitKey(5)
        if ch == 27:
            break
    cv2.destroyAllWindows()




