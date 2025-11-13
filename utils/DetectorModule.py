import numpy as np
import cv2
import os
from AlignerModule import Face_Aligner
from mtcnn import MTCNN
from utils import fix_box

class Face_Detector:
    def __init__(self, image_size=160, margin=10, detect_multiple_faces=False, min_face_size=20):
        self.aligner = Face_Aligner(desiredFaceWidth=image_size, margin=margin)
        self.detector = MTCNN()
        self.detect_multiple_faces = detect_multiple_faces

    def detect(self, img):
        bounding_boxes = self.detector.detect_faces(img)
        face_amount = len(bounding_boxes)
        if face_amount <= 0:
            return [], []
        det_arr = np.array([fix_box(b["box"]) for b in bounding_boxes])
        img_size = np.asarray(img.shape)[0:2]
        if face_amount > 1 and self.detect_multiple_faces == True:
            bounding_box_size = det_arr[:,2] * det_arr[:,3]
            img_center = img_size / 2
            offsets = np.vstack([det_arr[:, 0] + (det_arr[:, 2] / 2) - img_center[1],
                                 det_arr[:, 1] + (det_arr[:, 3] / 2) - img_center[0]])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            index = np.argmax(bounding_box_size - offset_dist_squared * 2.0)
            det = [det_arr[index, :]]
            landmarks = [bounding_boxes[index]['keypoints']]
        else:
            det = [np.squeeze(d) for d in det_arr]
            landmarks = [b['keypoints'] for b in bounding_boxes]
        return det, landmarks

    def extract(self, img):
        bounding_box, landmarks = self.detect(img)
        return [self.aligner.align(img, detection) for detection in zip(bounding_box, landmarks)]

def main(image_size=160, margin=10, detect_multiple_faces=False):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    input_dir = os.path.join(base_dir, "data", "train_data")
    output_dir = os.path.join(base_dir, "data", "output_data")


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    detector = Face_Detector(image_size, margin, detect_multiple_faces)

    for name in os.listdir(input_dir):
        person_input_dir = os.path.join(input_dir, name)
        person_output_dir = os.path.join(output_dir, name)

        if not os.path.isdir(person_input_dir):
            continue

        if not os.path.exists(person_output_dir):
            os.makedirs(person_output_dir)

        print(f"обработка для {name}:")
        counter = 0
        for img_name in os.listdir(person_input_dir):
            img_path = os.path.join(person_input_dir, img_name)

            if not os.path.isfile(img_path):
                continue

            img = cv2.imread(img_path)
            if img is None:
                print(f"Не удалось загрузить изображение: {img_path}")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            extracted_faces = detector.extract(img)

            if len(extracted_faces) == 0:
                print("Лиц не найдено")

            for i, face in enumerate(extracted_faces):
                suffix = ('_%d' % i) if detect_multiple_faces else ''
                name, ext = os.path.splitext(img_name)
                filename_output = f"{name}_{i}{ext}"
                output_filename = os.path.join(person_output_dir, filename_output)
                cv2.imwrite(output_filename, face)
            counter+=1
        print(f"Закончено, фотографий:{counter}")

    print("Обработка завершена")

if __name__ == "__main__":
    main()
