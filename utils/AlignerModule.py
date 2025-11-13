import os
import cv2
import math
import pandas as pd
from PIL import Image
import numpy as np

class Face_Aligner:
    def __init__(self, desiredLeftEye=(0.35, 0.35),
                 desiredFaceWidth=256, desiredFaceHeight=None, margin=0, alignment_method="affine_eyes"):
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.margin = margin
        self.alignment_method = alignment_method

        if desiredFaceHeight == None:
            self.desiredFaceHeight = desiredFaceWidth
        else:
            self.desiredFaceHeight = desiredFaceHeight

        self.ref_5points = np.array([
            [30.2946, 51.6963],  # левый глаз
            [65.5318, 51.5014],  # правый глаз
            [48.0252, 71.7366],  # нос
            [33.5493, 92.3655],  # левая часть рта
            [62.7299, 92.2041]  # правая часть рта
        ], dtype=np.float32)

        # нужный размер
        scale = self.desiredFaceWidth / 112.0
        self.ref_5points *= scale

    def align(self, image, detection, method=None):
        if method is None:
            method = self.alignment_method

        if method == "affine_5points":
            return self.align_affine_5points(image, detection)
        elif method == "perspective_5points":
            return self.align_perspective_5points(image, detection)
        elif method == "similarity":
            return self.align_similarity(image, detection)
        else:
            return self.align_affine_5points(image, detection)

    def align_affine_5points(self, image, detection): # афинное преобразование
        det, lm = detection

        src_points = np.array([
            lm['left_eye'],  # левый глаз
            lm['right_eye'],  # правый глаз
            lm['nose'],  # нос
            lm['mouth_left'],  # левый угол рта
            lm['mouth_right']  # правый угол рта
        ], dtype=np.float32)

        src_tri = src_points[:3]
        dst_tri = self.ref_5points[:3]

        M = cv2.getAffineTransform(src_tri, dst_tri) # Вычисляем матрицу аффинного преобразования

        # применение преобразования
        aligned = cv2.warpAffine(image, M,
                                 (self.desiredFaceWidth, self.desiredFaceHeight),
                                 flags=cv2.INTER_CUBIC)
        return aligned

    def align_perspective_5points(self, image, detection): # выравнивание c учетом перспективы
        det, lm = detection

        src_points = np.array([
            lm['left_eye'],
            lm['right_eye'],
            lm['nose'],
            lm['mouth_left'],
            lm['mouth_right']
        ], dtype=np.float32)

        # Используем матрицу перспективы
        M, mask = cv2.findHomography(src_points, self.ref_5points, cv2.RANSAC, 5.0)

        aligned = cv2.warpPerspective(image, M,
                                      (self.desiredFaceWidth, self.desiredFaceHeight),
                                      flags=cv2.INTER_CUBIC)
        return aligned

    def align_similarity(self, image, detection): # выравнивание с учетом симметрии
        det, lm = detection
        rightEyeCenter = lm['left_eye']
        leftEyeCenter = lm['right_eye']

        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]

        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.desiredFaceWidth
        scale = desiredDist / dist

        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
                      (leftEyeCenter[1] + rightEyeCenter[1]) // 2)
        eyesCenter = (int(eyesCenter[0]), int(eyesCenter[1]))
        M = cv2.getRotationMatrix2D(center=eyesCenter, angle=angle, scale=scale)

        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0] + self.margin)
        M[1, 2] += (tY - eyesCenter[1] + self.margin)

        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        aligned = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

        return aligned