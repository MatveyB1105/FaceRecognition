import numpy as np
import cv2
import os
import pandas as pd
from facenet_pytorch import InceptionResnetV1, MTCNN
import torch
from PIL import Image
import warnings
from config import DATASET_PATH, BASE_DIR, EMBEDDINGS_PATH, MY_EMBEDDINGS_PATH, MY_DATASET_PATH
from AlignerModule import Face_Aligner

warnings.filterwarnings('ignore')
import pickle


class Get_Embeddings:
    def __init__(self, model_name='facenet', use_detector=True, use_aligner=True):
        self.model_name = model_name
        self.use_detector = use_detector
        self.use_aligner = use_aligner

        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()

        self.detector = MTCNN(keep_all=True) if use_detector else None
        self.aligner = Face_Aligner() if use_aligner else None

    def extract_faces(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        boxes, probs, landmarks = self.detector.detect(pil_img, landmarks=True)

        if boxes is None:
            return []

        aligned_faces = []
        for i, (box, landmark) in enumerate(zip(boxes, landmarks)):
            x1, y1, x2, y2 = map(int, box)
            cropped_face = img[y1:y2, x1:x2]

            if self.use_aligner and self.aligner is not None:
                bbox = [x1, y1, x2 - x1, y2 - y1]
                face_landmarks = {
                    'left_eye': list(landmark[0]),
                    'right_eye': list(landmark[1]),
                    'nose': list(landmark[2]),
                    'mouth_left': list(landmark[3]),
                    'mouth_right': list(landmark[4])
                }
                try:
                    aligned_face = self.aligner.align(img, (bbox, face_landmarks))
                    aligned_faces.append(aligned_face)
                except Exception as e:
                    print(f"Ошибка при выравнивании лица: {e}")
                    aligned_faces.append(cropped_face)
            else:
                aligned_faces.append(cropped_face)

        return aligned_faces

    def get_embedding(self, face_img):
        # Если face_img - это путь к файлу
        if isinstance(face_img, str):
            img = cv2.imread(face_img)
            if img is None:
                return None
            face_img = img

        if isinstance(face_img, np.ndarray):
            face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_img_resized = cv2.resize(face_img_rgb, (160, 160))
            face_img_normalized = (face_img_resized - 127.5) / 128.0
            face_tensor = torch.tensor(face_img_normalized).permute(2, 0, 1).float().unsqueeze(0)
        else:
            face_tensor = face_img

        with torch.no_grad():
            embedding = self.resnet(face_tensor)

        return embedding.numpy().flatten()

    def process_directory(self, input_dir, output_pkl):
        embeddings = []
        labels = []
        image_paths = []

        for person_name in os.listdir(input_dir):
            person_dir = os.path.join(input_dir, person_name)

            if not os.path.isdir(person_dir):
                continue

            print(f"Обработка персоны: {person_name}")

            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)

                if not os.path.isfile(img_path):
                    continue

                img = cv2.imread(img_path)
                if img is None:
                    print(f"Не удалось загрузить изображение: {img_path}")
                    continue

                try:
                    if self.use_detector:
                        extracted_faces = self.extract_faces(img)

                        if len(extracted_faces) == 0:
                            print(f"Лица не обнаружены на изображении: {img_path}")
                            continue

                        embedding = self.get_embedding(extracted_faces[0])
                    else:

                        embedding = self.get_embedding(img)

                    embeddings.append(embedding)
                    labels.append(person_name)
                    image_paths.append(img_path)
                    print(f"Успешно обработано: {img_path}")

                except Exception as e:
                    print(f"Ошибка при обработке изображения {img_path}: {e}")
                    continue

        data = {
            'person': labels,
            'image_path': image_paths,
            'embedding': np.array(embeddings) if embeddings else np.array([])
        }

        os.makedirs(os.path.dirname(output_pkl), exist_ok=True)

        with open(output_pkl, 'wb') as f:
            pickle.dump(data, f)

        print(f"Эмбеддинги сохранены в: {output_pkl}")
        print(f"Обработано {len(labels)} изображений")

        return data


if __name__ == "__main__":
    embedder = Get_Embeddings(
        model_name='facenet',
        use_detector=True,
        use_aligner=True
    )

    embedder.process_directory(
        input_dir=MY_DATASET_PATH,
        output_pkl=MY_EMBEDDINGS_PATH
    )