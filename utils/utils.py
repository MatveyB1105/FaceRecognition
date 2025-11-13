import cv2
from PIL import Image
import numpy as np

def load_gray(file):
    img = np.asarray(Image.open(file).convert('L'))
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


def resize_img(img, img_size = None):
    if img_size == None:
        return img
    scaled = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
    scaled = scaled.reshape(-1, img_size, img_size, 3)
    return scaled
def fix_box(box):
    return [max(0,i) for i in box]