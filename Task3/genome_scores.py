import numpy as np
from preprocessing import *
import re
from functools import reduce
import pandas as pd


def preprocess(genome_scores):
    print(genome_scores.groupby(by=['movieId', 'tagId'].sort(key='relevance')))
