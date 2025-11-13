import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import time
from ClassifierModule import FaceClassifier
from config import MY_EMBEDDINGS_PATH
class FaceRecognizer:
    def __init__(self, classifier, threshold=0.7):
        self.classifier = classifier
        self.threshold = threshold
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def recognize_image(self, image_path, output_path=None):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")

        name, confidence, _ = self.classifier.predict(image_path, self.threshold)

        label = f"{name} ({confidence:.2f})"
        color = (0, 255, 0) if name != "Неизвестный" else (0, 0, 255)

        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        try:
            font = ImageFont.truetype("arial.ttf", 32)
        except:
            font = ImageFont.load_default()

        draw.text((10, 10), label, font=font, fill=color)

        img_with_text = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        if output_path:
            cv2.imwrite(output_path, img_with_text)

        return name, confidence, img_with_text

    def recognize_video(self, video_source=0, output_path=None):

        cap = cv2.VideoCapture(video_source)

        if not cap.isOpened():
            raise ValueError("Не удалось открыть видео источник")

        if output_path:
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        prev_time = 0
        fps = 0

        print("Нажмите 'q' для выхода, 's' для сохранения кадра")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = time.time()
            fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
            prev_time = current_time

            processed_frame = self.process_frame(frame)
            cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('Face Recognition', processed_frame)

            if output_path:
                out.write(processed_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = int(time.time())
                cv2.imwrite(f"frame_{timestamp}.jpg", processed_frame)
                print(f"Кадр сохранен как frame_{timestamp}.jpg")

        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        embedding = self.classifier.embedder.get_embedding(rgb_frame)

        if embedding is not None:
            distances, indices = self.classifier.classifier.kneighbors([embedding], n_neighbors=3)
            confidence = 1 - np.mean(distances)

            if confidence >= self.threshold:
                predicted_label = self.classifier.classifier.predict([embedding])[0]
                predicted_name = self.classifier.label_encoder.inverse_transform([predicted_label])[0]
            else:
                predicted_name = "Неизвестный"

            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w, h), (0, 255, 0), 2)

            label = f"{predicted_name} ({confidence:.2f})"
            color = (0, 255, 0) if predicted_name != "Неизвестный" else (0, 0, 255)

            cv2.putText(frame, label, (10, 30), self.font, 1, color, 2, cv2.LINE_AA)

        return frame

    def recognize_directory(self, directory_path, output_dir=None):
        results = {}

        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_paths = []

        for root, _, files in os.walk(directory_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_paths.append(os.path.join(root, file))

        print(f"Найдено {len(image_paths)} изображений для обработки")

        for i, image_path in enumerate(image_paths):
            print(f"Обработка {i + 1}/{len(image_paths)}: {os.path.basename(image_path)}")

            try:
                name, confidence, processed_img = self.recognize_image(image_path)

                results[image_path] = {
                    'name': name,
                    'confidence': confidence
                }

                if output_dir:
                    output_path = os.path.join(output_dir, os.path.basename(image_path))
                    cv2.imwrite(output_path, processed_img)

            except Exception as e:
                print(f"Ошибка при обработке {image_path}: {str(e)}")
                results[image_path] = {
                    'name': 'Ошибка',
                    'confidence': 0.0,
                    'error': str(e)
                }

        return results


if __name__ == "__main__":
    classifier = FaceClassifier(
        embeddings_path=MY_EMBEDDINGS_PATH,
        model_name='facenet',
        use_detector=True,
        use_aligner=True
    )

    classifier.train_classifier(test_size=0.2, n_neighbors=3)

    recognizer = FaceRecognizer(classifier, threshold=0.6)

    results = recognizer.recognize_video(
        video_source=0
    )
