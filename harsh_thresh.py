import cv2
import numpy as np
import os

def harsh_thresh(file):
    """
    A function that gets executed when ever the image is a screen shot.
    It creates a harsh thresholding, that isolates the document to some degree.
    """
    print(file)
    img = cv2.imread(file)
    img = cv2.threshold(img, 250, 255, cv2.THRESH_TOZERO)[1]
    cv2.imwrite('uploads/0_thresh.jpg', img)
    path = '0_thresh.jpg'
    return path