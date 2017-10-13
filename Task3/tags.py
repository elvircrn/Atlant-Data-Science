import pandas as pd
from preprocessing import *


def del_useless_tags(tags):
    del tags['tag']
    del tags['timestamp']
    del tags['userId']


def f():
    tags = pd.read_csv('Data/ratings_information/tags.csv')
    tags['tag_id'] = map_col_to_ind(
        pd.Series(data=tags['tag'].str.lower().astype(str).apply(lambda x: x.replace(' ', '')).astype(str), dtype=str))
