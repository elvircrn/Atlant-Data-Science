import numpy as np
import helpers as hlp
import data
import preprocess


def flip_faces(features):
    features = np.array(hlp.flatten(
        [features.tolist(),
         [np.flip(np.reshape(feature, (48, 48)), axis=1).tolist() for feature in features]]))
    features = np.array([np.array(feature).reshape(48 * 48) for feature in features])
    return features


def augment_emotion(emotion_groups, emo_id):
    return flip_faces(emotion_groups[emo_id][0]), np.tile(emotion_groups[emo_id][1], (2, 1)).reshape(-1, 8)


def augment_data(features, labels):
    emotion_groups = preprocess.split_emotions(features, labels)

    emotion_groups[data.DISGUST_ID] = augment_emotion(emotion_groups, data.DISGUST_ID)
    emotion_groups[data.FEAR_ID] = augment_emotion(emotion_groups, data.FEAR_ID)
    emotion_groups[data.CONTEMPT_ID] = augment_emotion(emotion_groups, data.CONTEMPT_ID)

    features = np.concatenate(tuple(group[0] for group in emotion_groups))
    labels = np.concatenate(tuple(group[1] for group in emotion_groups))
    return features, labels
