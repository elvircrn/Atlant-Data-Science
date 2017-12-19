import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def split_data(data, percentages):
    set_sizes = [int(percentage * len(data)) for percentage in percentages]
    set_sizes = np.cumsum(set_sizes)[:-1]
    sets = np.split(data, set_sizes)
    return sets


def get_cross_val(sets):
    return sets[2]


def delete_non_numeric(data):
    del data['Usage']
    del data['Image name']
    return data


def delete_weird(data):
    data = data[data['NF'] != 10]
    del data['NF']
    return data


def plot_emo_hist(data):
    data.hist(column=['emo_id'], cumulative='True')
    plt.savefig('Data/Graphs/emo_cnt.png')
    return data

def emotion_cnt(data):
    print(data)
    data['emo_id'] = data.idxmax()
    return data


def filter_secure(data):
    data['confidence'] = data.max(axis=1)
    return data


def plot_confidence_hist(data):
    data.hist(column=['confidence'])
    plt.savefig('Data/Graphs/confidence.png')
    return data


def plot_cum_confidence_hist(data):
    data.hist(column=['confidence'], cumulative='True')
    plt.savefig('Data/Graphs/cum_confidence.png')
    return data


def plot_confidence_cdf(data):
    data.hist(column=['confidence'], cumulative='True', bins=100, normed=1)
    plt.savefig('Data/Graphs/confidence_cdf.png')
    return data


def get_data():
    data = pd.read_csv('Data/FERPlus/fer2013new.csv')
    return data


def cross_val_analysis():
    data = (filter_secure
    (delete_non_numeric
     (delete_weird
      (get_cross_val
       (split_data
        (get_data(), [0.3, 0.3, 0.4]))))))

    plot_confidence_cdf(data)
    plot_confidence_hist(data)
    plot_cum_confidence_hist(data)

    return data


def emo_cnt_analysis():
    data = get_data()
    (plot_emo_hist
        (emotion_cnt
            (delete_non_numeric
                (delete_weird(data)))))
    return data


