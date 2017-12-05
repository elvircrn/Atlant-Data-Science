import tensorflow as tf
import numpy as np

import data
from preprocess import get_data
from helpers import perc_split

from tensorflow.contrib.learn.python.learn import monitors as monitor_lib


# yellow  - conv
# green   - max pool
# orange  - dropout
# blue    - fully connected
# gray    - soft-max
# def cnn_model_fn(features, labels, n_classes, dropout, reuse, is_training):
def cnn_model_fn(features, labels, mode):
    reuse = None
    dropout = 0.5
    n_classes = 8

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    with tf.variable_scope('ConvNet', reuse=reuse):
        # regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
        x = tf.reshape(features['images'], shape=[-1, 48, 48, 1])
        conv1 = tf.layers.conv2d(x, 64, 3, activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        mp1 = tf.layers.max_pooling2d(conv2, 2, 2)
        drop1 = tf.layers.dropout(mp1, rate=dropout, training=is_training)
        conv3 = tf.layers.conv2d(drop1, 128, 3, activation=tf.nn.relu)
        conv4 = tf.layers.conv2d(conv3, 128, 3, activation=tf.nn.relu)
        mp2 = tf.layers.max_pooling2d(conv4, 2, 2)
        drop2 = tf.layers.dropout(mp2, rate=dropout, training=is_training)
        conv5 = tf.layers.conv2d(drop2, 256, 3, activation=tf.nn.relu)
        conv6 = tf.layers.conv2d(conv5, 256, 3, activation=tf.nn.relu)
        conv7 = tf.layers.conv2d(conv6, 256, 3, activation=tf.nn.relu)
        mp3 = tf.layers.max_pooling2d(conv7, 2, 2)
        drop3 = tf.layers.dropout(mp3, rate=dropout, training=is_training)
        # conv8 = tf.layers.conv2d(drop3, 256, 3, activation=tf.nn.relu)
        # conv9 = tf.layers.conv2d(conv8, 256, 3, activation=tf.nn.relu)
        # conv10 = tf.layers.conv2d(conv9, 256, 3, activation=tf.nn.relu)
        # mp4 = tf.layers.max_pooling2d(conv10, 2, 2)
        # drop4 = tf.layers.dropout(mp4, rate=dropout, training=is_training)
        # fc1 = tf.contrib.layers.flatten(drop4)
        # fc1 = tf.layers.dense(fc1, 1024)
        # drop5 = tf.layers.dropout(fc1, rate=dropout, training=is_training)
        # fc2 = tf.contrib.layers.flatten(drop5)
        # fc2 = tf.layers.dense(fc2, 1024)
        # drop6 = tf.layers.dropout(fc2, rate=dropout, training=is_training)
        fc3 = tf.contrib.layers.flatten(drop3)
        # fc3 = tf.layers.dense(fc3, 9)
        out = tf.layers.dense(fc3, units=n_classes)

        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.argmax(input=out, axis=1),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            "probabilities": tf.nn.softmax(out, name="softmax_tensor")
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.PREDICT, predictions=predictions)

        epsilon = tf.constant(1e-8)
        out = out + epsilon

        # Calculate Loss (for both TRAIN and EVAL modes)
        # onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=n_classes)
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=labels, logits=out)

        tf.summary.scalar('Training Loss', loss)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                labels=labels, predictions=predictions["probabilities"])}

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


def predict(image):
    input_fn = tf.estimator.inputs.numpy_input_fn(x={'images': np.array(
        [image], dtype=np.float32)}, num_epochs=1, shuffle=False)
    model = tf.estimator.Estimator(cnn_model_fn, model_dir=data.MODEL_DIR)
    prediction = list(model.predict(input_fn))

    return prediction[0]['classes']


def get_validation_metrics():
    validation_metrics = {
        "accuracy":
            tf.contrib.learn.MetricSpec(
                metric_fn=tf.contrib.metrics.streaming_accuracy,
                prediction_key='accuracy')
    }

    return validation_metrics


def get_validation_monitor(test_data, test_labels, validation_metrics):
    validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
        test_data,
        test_labels,
        eval_steps=1,
        every_n_steps=50,
        metrics=validation_metrics)

    return validation_monitor


def launch_training():
    TRAINING_SET = 0
    TEST_SET = 1
    VALIDATION_SET = 2

    DATA = 0
    LABELS = 1

    datasets = get_data()

    batch_size = 64
    num_steps = 3000

    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'images': datasets[TRAINING_SET][DATA]}, y=datasets[TRAINING_SET][LABELS],
        batch_size=batch_size, num_epochs=None, shuffle=True)
    # Train the Model
    init = tf.global_variables_initializer()
    # saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        model = tf.estimator.Estimator(cnn_model_fn,
                                       model_dir=data.MODEL_DIR)

        validation_monitor = get_validation_monitor(datasets[TEST_SET][DATA], datasets[TEST_SET][LABELS],
                                                    get_validation_metrics())
        hooks = monitor_lib.replace_monitors_with_hooks([validation_monitor], model)
        model.train(input_fn, steps=num_steps)

        # model.train(input_fn, steps=num_steps)
