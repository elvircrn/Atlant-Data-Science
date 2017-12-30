import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn import ModeKeys
from tensorflow.contrib.learn import learn_runner

from serializer import Serializer
import preprocess
import data
import hyperopt
import architectures as arch

from hyperopt import hp


# Run only once
def initialize_flags():
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.app.flags.DEFINE_string(
        flag_name='model_dir', default_value=data.MODEL_DIR,
        docstring='Output directory for model and training stats.')


def get_flags():
    return tf.app.flags.FLAGS


def run_and_get_loss(params, run_config):
    runner = learn_runner.run(
        experiment_fn=experiment_fn,
        run_config=run_config,
        schedule="train_and_evaluate",
        hparams=params
    )
    return runner[0]['loss']


def get_run_config(model_id=None):
    if model_id is None:
        run_config = tf.contrib.learn.RunConfig(model_dir=get_flags().model_dir)
    else:
        run_config = tf.contrib.learn.RunConfig(model_dir=get_flags().model_dir + str(model_id))
    return run_config


def get_experiment_params():
    return tf.contrib.training.HParams(
        learning_rate=0.00000002,
        n_classes=data.N_CLASSES,
        train_steps=70000,
        min_eval_frequency=50,
        architecture=arch.padded_mini_vgg,
        dropout=0.7,
        validation=False
    )


def get_validation_params():
    return tf.contrib.training.HParams(
        learning_rate=0.0,
        n_classes=data.N_CLASSES,
        train_steps=1,
        min_eval_frequency=1,
        architecture=arch.padded_mini_vgg,
        dropout=1.0,
        validation=True
    )


def eager_hack():
    params = get_experiment_params()
    params.train_steps = 1
    run_and_get_loss(params, get_run_config())


mid = 0


def objective(args):
    # TODO: Refactor later
    global mid

    params = get_experiment_params()
    params.learning_rate = args['learn_rate']
    params.dropout = args['dropout']
    mid += 1
    run_config = get_run_config(model_id=mid)
    test_loss = run_and_get_loss(params, run_config)
    validation_loss = run_and_get_loss(get_validation_params(), run_config)

    print('Validation loss for model #{}: {}'.format(mid, validation_loss))

    return validation_loss


def run_experiment(argv=None):
    enable_hyperopt = argv[1]
    print('Hyper opt enabled: {}'.format(enable_hyperopt))

    if enable_hyperopt:
        space = {
            'learn_rate': hp.uniform('learn_rate', 0.000000001, 10.0),
            'dropout': hp.uniform('dropout', 0.4, 1.0)
        }

        best_model = hyperopt.fmin(objective, space, algo=hyperopt.tpe.suggest, max_evals=15)

        print(best_model)
        print(hyperopt.space_eval(space, best_model))
    else:
        params = get_experiment_params()
        run_config = tf.contrib.learn.RunConfig(model_dir=get_flags().model_dir)
        run_and_get_loss(params, run_config)


def experiment_fn(run_config, params):
    run_config = run_config.replace(
        save_checkpoints_steps=params.min_eval_frequency)
    estimator = get_estimator(run_config, params)

    # Setup data loaders
    datasets = Serializer.load_npy_datasets()
    train_input_fn, train_input_hook = get_train_inputs(batch_size=32, datasets=datasets)
    if params.validation:
        eval_input_fn, eval_input_hook = get_validation_inputs(batch_size=64, datasets=datasets)
    else:
        eval_input_fn, eval_input_hook = get_test_inputs(
            batch_size=128, datasets=datasets)

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


def get_estimator(run_config=None, params=None):
    if run_config is None:
        run_config = get_run_config()
    if params is None:
        params = get_experiment_params()

    return tf.estimator.Estimator(
        model_fn=model_fn,  # First-class function
        params=params,  # HParams
        config=run_config  # RunConfig
    )


def model_fn(features, labels, mode, params):
    is_training = mode == ModeKeys.TRAIN
    # Define model's architecture
    logits = params.architecture(inputs=features, dropout=params.dropout, is_training=is_training)
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
        # Hyper
        optimizer=tf.train.AdamOptimizer,
        learning_rate=params.learning_rate
    )


def f_score(predictions=None, labels=None, weights=None):
    p, update_op1 = tf.contrib.metrics.streaming_precision(predictions, labels)
    r, update_op2 = tf.contrib.metrics.streaming_recall(predictions, labels)
    eps = 1e-5
    return 2 * (p * r) / (p + r + eps), tf.group(update_op1, update_op2)


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
            dataset = dataset.shuffle(buffer_size=datasets[0][0].shape[0])
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
            dataset = dataset.shuffle(buffer_size=datasets[1][0].shape[0])
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


def get_validation_inputs(batch_size, datasets):
    iterator_initializer_hook = IteratorInitializerHook()

    def validation_inputs():
        with tf.name_scope(data.CV_SCOPE):
            images = datasets[2][0].reshape([-1, 48, 48, 1])
            labels = datasets[2][1]
            # Define placeholders
            images_placeholder = tf.placeholder(
                images.dtype, images.shape)
            labels_placeholder = tf.placeholder(
                labels.dtype, labels.shape)
            # Build dataset iterator
            dataset = tf.contrib.data.Dataset.from_tensor_slices(
                (images_placeholder, labels_placeholder))
            dataset = dataset.shuffle(buffer_size=700)
            dataset = dataset.batch(689)
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
    return validation_inputs, iterator_initializer_hook


def run_network(enable_gpu, enable_hyperopt):
    if enable_gpu:
        with tf.device("/gpu:0"):
            initialize_flags()
            tf.app.run(
                main=run_experiment,
                argv=[enable_gpu, enable_hyperopt]
            )
    else:
        initialize_flags()
        tf.app.run(
            main=run_experiment,
            argv=[enable_gpu, enable_hyperopt]
        )


def predict(estimator, images):
    images = np.reshape(images, [-1, 48, 48, 1]).astype(dtype=np.float32)
    predictions = estimator.predict(input_fn=lambda: images)
    return predictions
