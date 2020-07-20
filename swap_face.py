#!/usr/bin/python

import dlib
import numpy
from cv2 import cv2

IMG1_PATH = 'img1.jpg'
IMG2_PATH = 'img2.jpg'

#标记68个人脸关键点
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

class TooManyFaces(Exception):
    pass

class NoFaces(Exception):
    pass

def get_landmarks(img):
    face = detector(img, 1)

    if len(face) > 1:
        raise Exception('TooManyFaces')
    if len(face) == 0:
        raise Exception('NoFaces')

    return numpy.matrix([[p.x, p.y] for p in predictor(img, face[0]).parts()])

#人脸归一化处理(调整img2，使之符合img1)
def warp_img(points1, points2, img1, img2):
    points1 = points1.astype(numpy.float64)
    points2 = points2.astype(numpy.float64)

    c1 = numpy.mean(points1, axis=0)
    c2 = numpy.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = numpy.linalg.svd(points1.T * points2)
    R = (U * Vt).T

    transition_mat = numpy.vstack([numpy.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)), numpy.matrix([0., 0., 1.])])
    
    output_img = numpy.zeros(img1.shape, dtype=img2.dtype)
    cv2.warpAffine(img2, transition_mat[:2], (img1.shape[1], img1.shape[0]), dst=output_img, borderMode=cv2.BORDER_TRANSPARENT, flags=cv2.WARP_INVERSE_MAP)
    return output_img

#颜色调整
COLOUR_CORRECT_BLUR_FRAC = 0.6
LEFT_EYE_POINTS = list(range(42, 48))
RIGHT_EYE_POINTS = list(range(36, 42))

def correct_colours(im1, im2, landmarks1):
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * numpy.linalg.norm(numpy.mean(landmarks1[LEFT_EYE_POINTS], axis=0) - numpy.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)
    im2_blur += 128 * (im2_blur <= 1.0).astype(im2_blur.dtype)

    return (im2.astype(numpy.float64) * im1_blur.astype(numpy.float64) / im2_blur.astype(numpy.float64))

LEFT_EYE_POINTS = list(range(42, 48))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_BROW_POINTS = list(range(17, 22))
NOSE_POINTS = list(range(27, 35))
MOUTH_POINTS = list(range(48, 61))
OVERLAY_POINTS = [LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,NOSE_POINTS + MOUTH_POINTS,]
FEATHER_AMOUNT = 11

def draw_convex_hull(img, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(img, points, color=color)

#获取人脸掩膜
def get_face_mask(img, landmarks):
    img = numpy.zeros(img.shape[:2], dtype=numpy.float64)

    for group in OVERLAY_POINTS:
        draw_convex_hull(img, landmarks[group], color=1)

    img = numpy.array([img, img, img]).transpose((1, 2, 0))

    img = (cv2.GaussianBlur(img, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    img = cv2.GaussianBlur(img, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

    return img

#读取图片
img1 = cv2.imread(IMG1_PATH, cv2.IMREAD_COLOR)
img2 = cv2.imread(IMG2_PATH, cv2.IMREAD_COLOR)
#获取标记点
landmarks1 = get_landmarks(img1)
landmarks2 = get_landmarks(img2)
#获取人脸掩模
mask = get_face_mask(img2, landmarks2)
warped_mask = warp_img(landmarks1, landmarks2, img1, mask)
combined_mask = numpy.max([get_face_mask(img1, landmarks1), warped_mask], axis=0)

#调整img2大小颜色
img2 = warp_img(landmarks1, landmarks2, img1, img2)
img2 = correct_colours(img1, img2, landmarks1)
#合成图片
output_img = img1 * (1.0 - combined_mask) + img2 * combined_mask
#生成文件
cv2.imwrite('output.jpg', output_img)