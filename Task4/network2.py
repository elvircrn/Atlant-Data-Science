import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.learn import ModeKeys
from tensorflow.contrib.learn import learn_runner

import preprocess
import data


# Run only once in main
def initialize_flags():
    tf.logging.set_verbosity(tf.logging.DEBUG)
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_string(
        flag_name='model_dir', default_value=data.MODEL_DIR,
        docstring='Output directory for model and training stats.')


def get_flags():
    return tf.app.flags.FLAGS


def run_experiment(argv=None):
    params = tf.contrib.training.HParams(
        learning_rate=0.00002,
        n_classes=data.N_CLASSES,
        train_steps=50000,
        min_eval_frequency=50
    )

    run_config = tf.contrib.learn.RunConfig(model_dir=get_flags().model_dir)

    learn_runner.run(
        experiment_fn=experiment_fn,
        run_config=run_config,
        schedule="train_and_evaluate",
        hparams=params
    )


def experiment_fn(run_config, params):
    run_config = run_config.replace(
        save_checkpoints_steps=params.min_eval_frequency)
    estimator = get_estimator(run_config, params)
    # Setup data loaders
    datasets = preprocess.get_data(split_data=True)

    train_input_fn, train_input_hook = get_train_inputs(
        batch_size=64, datasets=datasets)
    eval_input_fn, eval_input_hook = get_test_inputs(
        batch_size=64, datasets=datasets)
    # Define the experiment
    experiment = tf.contrib.learn.Experiment(
        estimator=estimator,  # Estimator
        train_input_fn=train_input_fn,  # First-class function
        eval_input_fn=eval_input_fn,  # First-class function
        train_steps=params.train_steps,  # Mini-batch steps
        min_eval_frequency=params.min_eval_frequency,  # Eval frequency
        train_monitors=[train_input_hook],  # Hooks for training
        eval_hooks=[eval_input_hook],  # Hooks for evaluation
        eval_steps=None  # Use evaluation feeder until its empty
    )
    return experiment


def get_estimator(run_config, params):
    return tf.estimator.Estimator(
        model_fn=model_fn,  # First-class function
        params=params,  # HParams
        config=run_config  # RunConfig
    )


def model_fn(features, labels, mode, params):
    is_training = mode == ModeKeys.TRAIN
    # Define model's architecture
    logits = cnn_architecture(features, is_training=is_training)
    predictions = tf.argmax(logits, axis=1)
    # Loss, training and eval operations are not needed during inference.
    loss = None
    train_op = None
    eval_metric_ops = {}
    if mode != ModeKeys.INFER:
        loss = tf.losses.softmax_cross_entropy(
            labels,
            logits=logits)
        train_op = get_train_op_fn(loss, params)
        eval_metric_ops = get_eval_metric_ops(labels, predictions)
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops
    )


def get_train_op_fn(loss, params):
    return tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        optimizer=tf.train.AdamOptimizer,
        learning_rate=params.learning_rate
    )


def f_score(predictions=None, labels=None, weights=None):
    P, update_op1 = tf.contrib.metrics.streaming_precision(predictions, labels)
    R, update_op2 = tf.contrib.metrics.streaming_recall(predictions, labels)
    eps = 1e-5
    return 2 * (P * R) / (P + R + eps), tf.group(update_op1, update_op2)


def get_eval_metric_ops(labels, predictions):
    argmax_labels = tf.argmax(input=labels, axis=1)

    eval_dict = {
        'Accuracy': tf.metrics.accuracy(
            labels=argmax_labels,
            predictions=predictions,
            name='accuracy'),
        'Precision': tf.metrics.precision(
            labels=argmax_labels,
            predictions=predictions,
            name='precision'
        ),
        'Recall': tf.metrics.recall(
            labels=argmax_labels,
            predictions=predictions,
            name='recall'
        ),
        # 'F-Score': f_score(predictions, labels)
    }

    # precision = eval_dict['Precision'][0]
    # recall = eval_dict['Recall'][0]
    # f_score = (2 * precision * recall) / (precision + recall)
    # eval_dict['F-Score'] = f_score

    return eval_dict


# yellow  - conv
# green   - max pool
# orange  - dropout
# blue    - fully connected
# gray    - soft-max
# def cnn_model_fn(features, labels, n_classes, dropout, reuse, is_training):
def cnn_architecture(inputs, is_training, scope=data.DEFAULT_SCOPE):
    with tf.variable_scope(scope):
        with slim.arg_scope(
                [slim.conv2d, slim.fully_connected],
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                activation_fn=tf.nn.relu):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], padding='VALID', scope='conv1')
            net = slim.max_pool2d(net, 2, stride=2, scope='pool1')
            net = slim.dropout(net, is_training=is_training, scope='dropout1')

            net = slim.repeat(net, 3, slim.conv2d, 128, [3, 3], padding='VALID', scope='conv2')
            net = slim.max_pool2d(net, 2, stride=2, scope='pool2')
            net = slim.dropout(net, is_training=is_training, scope='dropout2')

            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], padding='VALID', scope='conv3')
            net = slim.max_pool2d(net, 2, stride=2, scope='pool3')
            net = slim.dropout(net, is_training=is_training, scope='dropout3')

            net = slim.flatten(net)
            net = slim.fully_connected(net, data.N_CLASSES, activation_fn=None, scope='fc1')

            net = slim.softmax(net, scope='sm1')
        return net


class IteratorInitializerHook(tf.train.SessionRunHook):
    """Hook to initialise data iterator after Session is created."""

    def __init__(self):
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_func = None

    def after_create_session(self, session, coord):
        """Initialise the iterator after the session has been created."""
        self.iterator_initializer_func(session)


def get_train_inputs(batch_size, datasets):
    iterator_initializer_hook = IteratorInitializerHook()

    def train_inputs():
        with tf.name_scope(data.TRAINING_SCOPE):
            images = datasets[0][0].reshape([-1, 48, 48, 1])
            labels = datasets[0][1]
            images_placeholder = tf.placeholder(
                images.dtype, images.shape)
            labels_placeholder = tf.placeholder(
                labels.dtype, labels.shape)
            # Build dataset iterator
            dataset = tf.contrib.data.Dataset.from_tensor_slices(
                (images_placeholder, labels_placeholder))
            dataset = dataset.repeat(None)  # Infinite iterations
            dataset = dataset.shuffle(buffer_size=10000)
            dataset = dataset.batch(batch_size)
            iterator = dataset.make_initializable_iterator()
            next_example, next_label = iterator.get_next()
            # Set run-hook to initialize iterator
            iterator_initializer_hook.iterator_initializer_func = \
                lambda sess: sess.run(
                    iterator.initializer,
                    feed_dict={images_placeholder: images,
                               labels_placeholder: labels})
            # Return batched (features, labels)
            return next_example, next_label

    # Return function and hook
    return train_inputs, iterator_initializer_hook


def get_test_inputs(batch_size, datasets):
    iterator_initializer_hook = IteratorInitializerHook()

    def test_inputs():
        with tf.name_scope(data.TEST_SCOPE):
            images = datasets[1][0].reshape([-1, 48, 48, 1])
            labels = datasets[1][1]
            # Define placeholders
            images_placeholder = tf.placeholder(
                images.dtype, images.shape)
            labels_placeholder = tf.placeholder(
                labels.dtype, labels.shape)
            # Build dataset iterator
            dataset = tf.contrib.data.Dataset.from_tensor_slices(
                (images_placeholder, labels_placeholder))
            dataset = dataset.batch(batch_size)
            iterator = dataset.make_initializable_iterator()
            next_example, next_label = iterator.get_next()
            # Set run-hook to initialize iterator
            iterator_initializer_hook.iterator_initializer_func = \
                lambda sess: sess.run(
                    iterator.initializer,
                    feed_dict={images_placeholder: images,
                               labels_placeholder: labels})
            return next_example, next_label

    # Return function and hook
    return test_inputs, iterator_initializer_hook


def run_network():
    initialize_flags()
    tf.app.run(
        main=run_experiment
    )
