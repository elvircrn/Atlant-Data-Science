import numpy as np 
import matplotlib as mp
import matplotlib.pyplot as plt
import tensorflow as tf
import math
import architectures as arch


def get_activations(layer, stimuli):
    with tf.Session(graph=layer.graph) as sess:
        sess.run(tf.global_variables_initializer())
        units = sess.run(layer, feed_dict={arch.get_inputs(): np.reshape(stimuli, [1, 48, 48, 1], order='F')})
        plot_nnfilter(units)


def plot_nnfilter(units):
    filters = units.shape[3]
    plt.figure(1, figsize=(48, 48))
    n_columns = 6
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i + 1)
        plt.imshow(units[0, :, :, i], interpolation='nearest', cmap='gray')

