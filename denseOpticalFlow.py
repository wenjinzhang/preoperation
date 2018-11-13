import cv2
import numpy as np


def convert(fold_name):
    cap = cv2.VideoCapture(fold_name+"/%2d.png")
    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    count = 0
    while(1):
        count = count + 1
        ret, frame2 = cap.read()
        if not ret:
            break
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs,  # prevImg
                                            next,  # nextImg
                                            None,  # prevPts
                                            0.5,  # nextPts
                                            3,  # status
                                            15,  # err
                                            2,  # winSize
                                            5,  # maxLevel
                                            1.2,  # criteria
                                            0)  # flags

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow('frame2', rgb)
        cv2.imwrite("{}out/{:0>4d}.png".format(fold_name, count), rgb)
        prvs = next
        count += 1
    cap.release()

if __name__=="__main__":
    convert('imgs')





