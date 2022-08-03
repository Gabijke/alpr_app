import numpy as np
import math
import cv2
import numpy as np


def rotate_image(image: np.array, angle: float) -> np.array:
    
    if len(src_img.shape) == 3:
        h, w, _ = src_img.shape
    else:
        h, w = src_img.shape
        
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def compute_skew(src_img: np.array) -> float:

    img = cv2.medianBlur(src_img, 3)
    edges = cv2.Canny(img,  threshold1=30,  threshold2=100, apertureSize=3, L2gradient=True)
    lines = cv2.HoughLinesP(edges, 1, math.pi/180, 30, minLineLength= w/4.0, maxLineGap= h/4.0)
    angle = 0.0

    cnt = 0

    try:
        for x1, y1, x2, y2 in lines[0]:
            ang = np.arctan2(y2 - y1, x2 - x1)
            if math.fabs(ang) <= 30:
                angle += ang
                cnt += 1
    except TypeError:
        return 0

    if cnt == 0:
        return 0.0
    return (angle / cnt)*180/math.pi


def deskew_img(src_img: np.array):
    return rotate_image(src_img, compute_skew(src_img))
