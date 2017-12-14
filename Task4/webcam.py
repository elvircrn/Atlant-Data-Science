import cv2
import numpy as np
import tensorflow as tf

from facedetect import detect

import visualize as vz
import network2


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

    network2.initialize_flags()
    estimator = network2.get_estimator()

    while rval:
        rval, frame = vc.read()
        frame = cv2.flip(frame, 1)
        full_frame, images, lower_right_corners, extracted_faces = detect(frame)

        if len(images) > 0:
            if enable_predictions:
                predictions = network2.predict(estimator, images)
            else:
                predictions = [0]

            vis = np.concatenate(extracted_faces, axis=1)
            cv2.imshow(EXTRACTED_WINDOW, vis)

            # TODO: Add layer visualization here
            # vz.get_activations(

            for prediction, lower_right_corner in zip(predictions, lower_right_corners):
                cv2.putText(frame, labels[prediction],
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

