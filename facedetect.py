import numpy as np
import cv2


def flatten(image_matrix):
    return [item for sublist in image_matrix for item in sublist]


def greyscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def load_img(img_path):
    return cv2.imread(img_path)


class FaceExtractor:
    face_cascade = None

    @staticmethod
    def get_cascade():
        if FaceExtractor.face_cascade is None:
            FaceExtractor.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        return FaceExtractor.face_cascade

    @staticmethod
    def extract_face(image):
        gray = greyscale(image)
        (x, y, w, h) = FaceExtractor.get_cascade().detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )[0]
        image = cv2.rectangle(image, (x + 1, y + 1), (x + w - 1, y + h - 1), (0, 255, 0), 2)
        crop_img = image[y: y + h, x: x + w]
        crop_img = cv2.resize(crop_img, (48, 48), cv2.INTER_CUBIC)
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        return np.asarray(flatten(crop_img), dtype=np.float32)

    @staticmethod
    def extract_faces(image):
        gray = greyscale(image)

        faces = FaceExtractor.get_cascade().detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        images = []
        lower_right_corners = []
        full_faces = []
        face_features = []

        for (x, y, w, h) in faces:
            crop_img = image[y: y + h, x: x + w]
            image = cv2.rectangle(image, (x + 1, y + 1), (x + w - 1, y + h - 1), (255, 0, 0), 2)
            full_faces.append(crop_img)
            crop_img = cv2.resize(crop_img, (48, 48), cv2.INTER_CUBIC)
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            images.append(cv2.resize(crop_img, (128, 128), cv2.INTER_CUBIC))
            lower_right_corners.append((y + h, x + w))
            face_features.append(flatten(crop_img))

        return face_features, lower_right_corners, full_faces, images, image

