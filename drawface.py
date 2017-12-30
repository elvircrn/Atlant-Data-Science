import matplotlib.pyplot as plt
import numpy as np


def draw_face(face):
    plt.imshow(np.reshape(face, (48, 48)))
