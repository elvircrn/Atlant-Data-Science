import numpy as np
import helpers as hlp


def flip_faces(features):
    features = np.array(hlp.flatten([features.tolist(), [np.flip(feature, axis=1).tolist() for feature in features]]))
    return features
