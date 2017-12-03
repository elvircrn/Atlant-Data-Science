import numpy as np
import cv2


def flatten(image_matrix):
    return [item for sublist in image_matrix for item in sublist]


def detect(image=None):
    if image is None:
        image = cv2.imread('Data/ellen.jpg')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # print("Found {0} faces!".format(len(faces)))

    images = []
    lower_right_corners = []

    ind = 0
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        image = cv2.rectangle(image, (x + 1, y + 1), (x + w - 1, y + h - 1), (0, 255, 0), 2)
        crop_img = image[y: y + h, x: x + w]  # Crop from x, y, w, h -> 100, 200, 300, 400
        ind += 1
        crop_img = cv2.resize(crop_img, (48, 48), cv2.INTER_CUBIC)
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite("face{}.jpg".format(ind), crop_img)
        images.append(crop_img)
        lower_right_corners.append((y + h, x + w))

    # cv2.imshow("Faces found", image)

    return image, [flatten(img) for img in images], lower_right_corners, images

