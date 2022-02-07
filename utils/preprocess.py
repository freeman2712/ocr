import cv2
import numpy as np


def resizeImage(img, scale = 0.75):
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)

    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)



def toGrayScale(img):
    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
