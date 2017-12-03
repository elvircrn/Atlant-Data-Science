import cv2
from facedetect import detect

import network


class FontData:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (255, 255, 255)
    line_type = 2


def launch_webcam():
    labels = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']

    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)
    # try to get the first frame
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False

    enable_predictions = False

    while rval:
        rval, frame = vc.read()
        full_frame, images, lower_right_corners, extracted_faces = detect(frame)

        if len(images) > 0:
            if enable_predictions:
                predictions = [network.predict(image) for image in images]
            else:
                predictions = [0]
            cv2.imshow("preview2", extracted_faces[0])
            for prediction, lower_right_corner in zip(predictions, lower_right_corners):
                cv2.putText(frame, labels[prediction],
                            tuple(reversed(lower_right_corner)),
                            FontData.font,
                            FontData.font_scale,
                            FontData.font_color,
                            FontData.line_type)
        cv2.imshow("preview", full_frame)
        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break

    cv2.destroyWindow("preview")
