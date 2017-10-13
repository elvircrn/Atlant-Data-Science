import pandas as pd
import numpy as np


def set_indices(links):
    links.set_index('movieId')
    links.set_index('imdbId')
    links.set_index('tmdbId')
    return links


def fill_na(links):
    it = -1
    for link in links['tmdbId']:
        if pd.isnull(link):
            link = --it
    return links


def pad_imdb(links):
    links['imdb_str'] = links['imdbId'].apply(lambda x: str(x).zfill(7)).astype(str)
    links['imdbId'] = links['imdb_str']
    del links['imdb_str']
    return links


def preprocess(links):
    return (set_indices
            (fill_na
             (pad_imdb(links))))
