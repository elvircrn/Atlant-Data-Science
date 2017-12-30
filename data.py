import os

# TODO: Start using relative path
MODEL_DIR = os.path.abspath("Data/log")

N_CLASSES = 8
N_VOTES = 10

DEFAULT_SCOPE = 'EmoConv'

TRAINING_SCOPE = 'Training_data'
TEST_SCOPE = 'Test_data'

LABELS = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']

FEATURES_FILE = 'Serialized/features.npy'
LABELS_FILE = 'Serialized/labels.npy'

MAIN_WINDOW_NAME = "Emotion"
EXTRACTED_WINDOW_NAME = "Extracted faces"

NEUTRAL_ID = 0
HAPPINESS_ID = 1
SURPRISE_ID = 2
SADNESS_ID = 3
ANGER_ID = 4
DISGUST_ID = 5
FEAR_ID = 6
CONTEMPT_ID = 7

