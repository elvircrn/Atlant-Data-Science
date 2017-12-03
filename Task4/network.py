import tensorflow as tf
import numpy as np
from facedetect import detect
from preprocess import get_train


# yellow  - conv
# green   - max pool
# orange  - dropout
# blue    - fully connected
# gray    - soft-max
# def cnn_model_fn(features, labels, n_classes, dropout, reuse, is_training):
def cnn_model_fn(features, labels, mode):
    reuse = None
    dropout = 0.5
    n_classes = 9
    with tf.variable_scope('ConvNet', reuse=reuse):
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
        out = tf.layers.dense(fc3, n_classes)

        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.argmax(input=out, axis=1),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            "probabilities": tf.nn.softmax(out, name="softmax_tensor")
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.PREDICT, predictions=predictions)

        mode = tf.estimator.ModeKeys.TRAIN

        # Calculate Loss (for both TRAIN and EVAL modes)
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=9)
        # loss = tf.losses.softmax_cross_entropy(
        # onehot_labels=onehot_labels, logits=predictions['probabilities'])
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=labels, logits=predictions['probabilities'])

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            tf.summary.scalar('Training Loss', loss)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                labels=labels, predictions=predictions["classes"])}
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def predict(image):
    input_fn = tf.estimator.inputs.numpy_input_fn(x={'images': np.array(
      [image], dtype=np.float32)}, num_epochs=1, shuffle=False)
    model = tf.estimator.Estimator(cnn_model_fn, model_dir='C:\\Users\\elvircrn\\Documents\\codes2\\Atlant Data Science\\Project1\\Atlant-Data-Science\\Task4\\Data\\log')
    prediction = list(model.predict(input_fn))

    return prediction[0]['classes']


def launch_training():
    train, labels = get_train()
    tr = tf.convert_to_tensor(train, dtype=tf.float32)
    lb = tf.convert_to_tensor(labels, dtype=tf.float32)

    batch_size = 20
    num_steps = 3000

    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'images': train}, y=labels,
        batch_size=batch_size, num_epochs=None, shuffle=True)
    # Train the Model
    init = tf.global_variables_initializer()
    # saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        model = tf.estimator.Estimator(cnn_model_fn,
                                       model_dir='C:\\Users\\elvircrn\\Documents\\codes2\\Atlant Data Science\\Project1\\Atlant-Data-Science\\Task4\\Data\\log')
        model.train(input_fn, steps=num_steps)
        # Add ops to save and restore all the variables.
