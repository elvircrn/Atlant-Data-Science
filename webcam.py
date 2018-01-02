import cv2
import numpy as np
import tensorflow as tf
import facedetect as fd
import visualize as vz
import network2
import architectures
from facedetect import FaceExtractor
# import predictionmanager as pm
import asyncio
import data


class FontData:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (255, 255, 255)
    line_type = 2


def async_webcam():
    loop = asyncio.get_event_loop()
    queue = asyncio.LifoQueue(loop=loop, maxsize=1)
    producer_coro = produce_faces(queue)
    consumer_coro = consume_faces(queue)
    # asyncio.coroutine(producer_coro)
    # asyncio.coroutine(consumer_coro)

    loop.run_until_complete(consumer_coro)

    loop.close()
    cv2.namedWindow(data.MAIN_WINDOW_NAME)
    # noinspection PyArgumentList
    vc = cv2.VideoCapture(0)

    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        rval, frame = vc.read()
        full_frame = cv2.flip(frame, 1)
        cv2.imshow(data.MAIN_WINDOW_NAME, full_frame)
        images, lower_right_corners, extracted_faces = FaceExtractor.extract_faces(full_frame)

        if len(images) > 0 and queue.qsize() < queue.maxsize:
            queue.put_nowait((images, extracted_faces))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyWindow(data.MAIN_WINDOW_NAME)


async def consume_faces(queue):
    network2.initialize_flags()
    estimator = network2.get_estimator()
    while True:
        features, faces = await queue.get()
        predictions = network2.predict(estimator, features)
        for prediction, face in zip(predictions, faces):
            cv2.putText(face,
                        data.LABELS[prediction],
                        (15, 15),
                        FontData.font,
                        FontData.font_scale,
                        FontData.font_color,
                        FontData.line_type)
            cv2.imshow(data.EXTRACTED_WINDOW_NAME, face)
        queue.task_done()


async def produce_faces(queue):
    cv2.namedWindow(data.MAIN_WINDOW_NAME)
    # noinspection PyArgumentList
    vc = cv2.VideoCapture(0)

    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        rval, frame = vc.read()
        full_frame = cv2.flip(frame, 1)
        cv2.imshow(data.MAIN_WINDOW_NAME, full_frame)
        images, lower_right_corners, extracted_faces = FaceExtractor.extract_faces(full_frame)

        if len(images) > 0 and queue.qsize() < queue.maxsize:
            queue.put_nowait((images, extracted_faces))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyWindow(data.MAIN_WINDOW_NAME)


def launch_webcam():
    cv2.namedWindow(data.MAIN_WINDOW_NAME)
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

            if len(extracted_faces) == 0:
                continue

            vis = np.concatenate(extracted_faces, axis=1)
            cv2.imshow(data.EXTRACTED_WINDOW_NAME, vis)

            if enable_visualization:
                vz.get_activations(architectures.get_layers()[0], images[0])

            for prediction, lower_right_corner in zip(predictions, lower_right_corners):
                label = data.LABELS[prediction]
                cv2.putText(full_frame, label,
                            tuple(reversed(lower_right_corner)),
                            FontData.font,
                            FontData.font_scale,
                            FontData.font_color,
                            FontData.line_type)
                cv2.imshow(data.MAIN_WINDOW_NAME, full_frame)
        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break

    cv2.destroyWindow(data.MAIN_WINDOW_NAME)
