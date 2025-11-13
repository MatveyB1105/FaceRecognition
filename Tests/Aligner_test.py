import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from utils import Face_Aligner
from mtcnn import MTCNN


class AlignerTester:
    def __init__(self, test_image_path):
        self.test_image_path = test_image_path
        self.image = cv2.imread(test_image_path)
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.detector = MTCNN()

    def detect_face(self):
        detections = self.detector.detect_faces(self.image_rgb)

        main_face = detections[0]
        box = main_face['box']
        keypoints = main_face['keypoints']

        print(f"Обнаружено лиц: {len(detections)}")
        print(f"Уверенность: {main_face['confidence']:.3f}")

        det = [box[0], box[1], box[2], box[3]]  # x, y, width, height

        landmarks = {
            'left_eye': list(keypoints['left_eye']),
            'right_eye': list(keypoints['right_eye']),
            'nose': list(keypoints['nose']),
            'mouth_left': list(keypoints['mouth_left']),
            'mouth_right': list(keypoints['mouth_right'])
        }

        print("Landmarks от MTCNN:")
        for name, point in landmarks.items():
            print(f"  {name}: {point}")

        return (det, landmarks)

    def draw_landmarks(self, detection):
        det, landmarks = detection

        img_viz = self.image_rgb.copy()

        x, y, w, h = det
        cv2.rectangle(img_viz, (int(x), int(y)), (int(x + w), int(y + h)), (255, 255, 0), 2)

        colors = {
            'left_eye': (255, 0, 0),  # Красный
            'right_eye': (0, 255, 0),  # Зеленый
            'nose': (0, 0, 255),  # Синий
            'mouth_left': (255, 255, 0),  # Голубой
            'mouth_right': (255, 0, 255)  # Розовый
        }

        for name, point in landmarks.items():
            color = colors[name]
            cv2.circle(img_viz, (int(point[0]), int(point[1])), 6, color, -1)
            cv2.putText(img_viz, name, (int(point[0]) + 5, int(point[1]) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.line(img_viz,
                 (int(landmarks['left_eye'][0]), int(landmarks['left_eye'][1])),
                 (int(landmarks['right_eye'][0]), int(landmarks['right_eye'][1])),
                 (255, 255, 255), 1)
        cv2.line(img_viz,
                 (int(landmarks['nose'][0]), int(landmarks['nose'][1])),
                 (int(landmarks['left_eye'][0]), int(landmarks['left_eye'][1])),
                 (255, 255, 255), 1)
        cv2.line(img_viz,
                 (int(landmarks['nose'][0]), int(landmarks['nose'][1])),
                 (int(landmarks['right_eye'][0]), int(landmarks['right_eye'][1])),
                 (255, 255, 255), 1)

        return img_viz

    def test_alignment_methods(self):
        detection = self.detect_face()
        methods = ["affine_5points", "perspective_5points", "similarity"]
        results = {}

        for method in methods:
            try:
                # Создаем aligner с параметрами
                aligner = Face_Aligner(
                    desiredFaceWidth=256,
                    desiredFaceHeight=256,
                    desiredLeftEye=(0.35, 0.35),
                    alignment_method=method
                )

                aligned_face = aligner.align(self.image, detection, method)
                results[method] = aligned_face
                print(f" {method}: успешно")

            except Exception as e:
                print(f" {method}: ошибка - {e}")
                results[method] = None

        return results, detection

    def evaluate_alignment_quality(self, results):
        quality_metrics = {}

        for method, aligned_img in results.items():
            if aligned_img is None:
                continue

            gray = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2GRAY)

            metrics = {
                'brightness': np.mean(aligned_img),
                'contrast': np.std(aligned_img),
                'sharpness': cv2.Laplacian(gray, cv2.CV_64F).var(),
                'entropy': self.calculate_entropy(gray),
                'face_symmetry': self.calculate_symmetry(aligned_img)
            }

            quality_metrics[method] = metrics

            print(f"\n {method}:")
            print(f"   Яркость: {metrics['brightness']:.1f}")
            print(f"   Контраст: {metrics['contrast']:.1f}")
            print(f"   Резкость: {metrics['sharpness']:.1f}")
            print(f"   Энтропия: {metrics['entropy']:.3f}")
            print(f"   Симметрия: {metrics['face_symmetry']:.3f}")

        return quality_metrics
    def calculate_entropy(self, image):
        histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
        histogram = histogram / histogram.sum()
        entropy = -np.sum(histogram * np.log2(histogram + 1e-10))
        return entropy
    def calculate_symmetry(self, face_image):
        height, width = face_image.shape[:2]
        mid = width // 2

        left_half = face_image[:, :mid]
        right_half = face_image[:, mid:]

        right_half_flipped = cv2.flip(right_half, 1)

        left_hist = cv2.calcHist([left_half], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        right_hist = cv2.calcHist([right_half_flipped], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

        cv2.normalize(left_hist, left_hist)
        cv2.normalize(right_hist, right_hist)

        similarity = cv2.compareHist(left_hist, right_hist, cv2.HISTCMP_CORREL)
        return max(0, similarity)
    def visualize_comparison(self, results, quality_metrics, detection):
        fig = plt.figure(figsize=(16, 12))

        plt.subplot(2, 3, 1)
        lena_with_marks = self.draw_landmarks(detection)
        plt.imshow(lena_with_marks)
        detector_name = "MTCNN" if self.detector is not None else "Ручные координаты"
        plt.title(f'Исходное изображение Лены\nДетектор: {detector_name}', fontsize=12)
        plt.axis('off')

        methods = list(results.keys())
        for idx, method in enumerate(methods, 2):
            plt.subplot(2, 3, idx)

            if results[method] is not None:
                aligned_rgb = cv2.cvtColor(results[method], cv2.COLOR_BGR2RGB)
                plt.imshow(aligned_rgb)

                if method in quality_metrics:
                    metrics = quality_metrics[method]
                    title = (f'{method}\n'
                             f'Резкость: {metrics["sharpness"]:.0f} | '
                             f'Симметрия: {metrics["face_symmetry"]:.2f}')
                else:
                    title = method

                plt.title(title, fontsize=11)
            else:
                plt.text(0.5, 0.5, 'Ошибка\nвыравнивания',
                         ha='center', va='center', transform=plt.gca().transAxes)
                plt.title(method, fontsize=11)

            plt.axis('off')

        plt.tight_layout()
        plt.savefig('lena_alignment_comparison_mtcnn.jpg', dpi=150, bbox_inches='tight')
        plt.show()

    def run_complete_analysis(self):
        print("Анализ методов выравнивания лиц")

        results, detection = self.test_alignment_methods()

        # Оцениваем качество
        quality_metrics = self.evaluate_alignment_quality(results)

        # Визуализируем результаты
        self.visualize_comparison(results, quality_metrics, detection)

        # Сохраняем результаты
        self.save_alignment_results(results)

        return results, quality_metrics, detection

    def save_alignment_results(self, results):
        output_dir = "lena_alignment_results_mtcnn"
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nСохранение результатов в папку '{output_dir}':")

        for method, aligned_img in results.items():
            if aligned_img is not None:
                filename = os.path.join(output_dir, f"lena_{method}.jpg")
                cv2.imwrite(filename, aligned_img)
                print(f"   ✅ {method} -> {filename}")



if __name__ == "__main__":
    tester = AlignerTester('test_face.png')
    results, metrics, detection = tester.run_complete_analysis()

