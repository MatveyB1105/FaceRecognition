import numpy as np
from collections import Counter
import cv2
import os
import pickle

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from Get_Embedding import Get_Embeddings
from pathlib import Path
from config import DATASET_PATH, BASE_DIR, EMBEDDINGS_PATH, TEST_PATH, MY_EMBEDDINGS_PATH


class FaceClassifier:
    def __init__(self, embeddings_path, model_name='facenet', use_detector=True, use_aligner=True, classifier_name='RandomForest'):
        self.embeddings_path = embeddings_path
        self.model_name = model_name
        self.use_detector = use_detector
        self.use_aligner = use_aligner

        self.load_embeddings()

        self.embedder = Get_Embeddings(
            model_name=model_name,
            use_detector=use_detector,
            use_aligner=use_aligner
        )
        self.classifier_name = classifier_name
        self.classifier = None
        self.label_encoder = LabelEncoder()

    def load_embeddings(self):
        with open(self.embeddings_path, 'rb') as f:
            data = pickle.load(f)

        self.embeddings = data['embedding']
        self.labels = data['person']
        self.image_paths = data['image_path']

        print(f"Количество эмбеддингов: {len(self.embeddings)}, Количество человек: {len(set(self.labels))}")

    def train_classifier(self, test_size=0.2, n_neighbors=3, min_samples=2):
        label_counts = Counter(self.labels)
        valid_labels = [label for label, count in label_counts.items() if count >= min_samples]

        if len(valid_labels) < 2:
            raise ValueError(
                f"Недостаточно классов с минимальным количеством образцов ({min_samples}). Число подходящих классов: {len(valid_labels)}")

        valid_indices = [i for i, label in enumerate(self.labels) if label in valid_labels]
        filtered_embeddings = self.embeddings[valid_indices]
        filtered_labels = [self.labels[i] for i in valid_indices]
        print("Фильтрация...")
        print(f"Количество подходящих эмбеддингов: {len(filtered_embeddings)}, количество классов: {len(valid_labels)}")

        encoded_labels = self.label_encoder.fit_transform(filtered_labels)

        X_train, X_test, y_train, y_test = train_test_split(
            filtered_embeddings, encoded_labels,
            test_size=test_size,
            random_state=42,
            stratify=encoded_labels
        )

        if self.classifier_name == "KNN":
            self.classifier = KNeighborsClassifier(
                n_neighbors=min(n_neighbors, len(X_train)),
                metric='cosine'
            )
        elif self.classifier_name == "SVM":
            self.classifier = SVC(kernel='linear', probability=True)
        elif self.classifier_name == "XGBoost":
            self.classifier = XGBClassifier(
                eval_metric='mlogloss',
                use_label_encoder=False,
                max_depth=3,
                n_estimators=100,
                random_state=42
            )
        else:
            model = RandomForestClassifier(n_estimators=1000, max_leaf_nodes=100, n_jobs=-1)

        self.classifier.fit(X_train, y_train)

        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Accuracy of training dataset: {accuracy:.3f}")

        return accuracy

    def predict(self, image_path, threshold=0.7):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")

        embedding = self.embedder.get_embedding(img)
        if embedding is None:
            return "Неизвестный", 0.0, None

        distances, indices = self.classifier.kneighbors([embedding], n_neighbors=3)

        confidence = 1 - np.mean(distances)

        if confidence < threshold:
            return "Неизвестный", confidence, embedding

        predicted_label = self.classifier.predict([embedding])[0]
        predicted_name = self.label_encoder.inverse_transform([predicted_label])[0]

        return predicted_name, confidence, embedding

    def add_new_person(self, image_paths, person_name, update_classifier=True):

        new_embeddings = []

        for img_path in image_paths:
            embedding = self.embedder.get_embedding(img_path)
            if embedding is not None:
                new_embeddings.append(embedding)

        if not new_embeddings:
            print("Не удалось извлечь эмбеддинги для нового человека")
            return False

        self.embeddings = np.vstack([self.embeddings, new_embeddings])
        self.labels.extend([person_name] * len(new_embeddings))

        if update_classifier:
            encoded_labels = self.label_encoder.fit_transform(self.labels)
            self.classifier.fit(self.embeddings, encoded_labels)

        print(f"Добавлен новый человек: {person_name} с {len(new_embeddings)} эмбеддингами")
        return True

    def evaluate_on_dataset(self, test_dir):
        true_labels = []
        pred_labels = []

        for person_name in os.listdir(test_dir):
            person_dir = os.path.join(test_dir, person_name)

            if not os.path.isdir(person_dir):
                continue

            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)

                # Предсказываем для каждого изображения
                predicted_name, confidence, _ = self.predict(img_path)

                true_labels.append(person_name)
                pred_labels.append(predicted_name)

        accuracy = accuracy_score(true_labels, pred_labels)
        print(f"Точность на тестовом наборе: {accuracy:.2f}")

        return accuracy

if __name__ == "__main__":
    classifier = FaceClassifier(
        embeddings_path=MY_EMBEDDINGS_PATH,
        model_name='facenet',
        use_detector=True,
        use_aligner=True,
        classifier_name='XGBoost'
    )
    classifier.train_classifier(test_size=0.2, n_neighbors=3)

# SVM: accuracy - 0.852
# KNN: accuracy - 0.910
# XGBoost: accuracy - ?
