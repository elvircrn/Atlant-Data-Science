import cv2
import numpy as np
import tensorflow as tf

import facedetect as fd

import visualize as vz
import network2
import architectures

from facedetect import FaceExtractor


class FontData:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (255, 255, 255)
    line_type = 2


def launch_webcam():
    MAIN_WINDOW_NAME = "Emotion"
    EXTRACTED_WINDOW = "Extracted faces"

    labels = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']

    cv2.namedWindow(MAIN_WINDOW_NAME)
    vc = cv2.VideoCapture(0)
    # try to get the first frame
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False

    enable_predictions = True
    enable_visualization = False

    network2.initialize_flags()
    estimator = network2.get_estimator()

    if enable_visualization:
        network2.eager_hack()

    while rval:
        rval, frame = vc.read()
        full_frame = cv2.flip(frame, 1)
        images, lower_right_corners, extracted_faces = FaceExtractor.extract_faces(full_frame)

        if len(images) > 0:
            if enable_predictions:
                predictions = network2.predict(estimator, images)
            else:
                predictions = np.zeros(len(images))

            vis = np.concatenate(extracted_faces, axis=1)
            cv2.imshow(EXTRACTED_WINDOW, vis)

            if enable_visualization:
                vz.get_activations(architectures.get_layers()[0], images[0])

            for prediction, lower_right_corner in zip(predictions, lower_right_corners):
                # label = labels[prediction] if np.random.rand() < 0.8 else 'ugly'
                label = labels[prediction]
                cv2.putText(full_frame, label,
                            tuple(reversed(lower_right_corner)),
                            FontData.font,
                            FontData.font_scale,
                            FontData.font_color,
                            FontData.line_type)
        cv2.imshow(MAIN_WINDOW_NAME, full_frame)
        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break

    cv2.destroyWindow(MAIN_WINDOW_NAME)
