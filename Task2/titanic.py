import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from mpl_toolkits.mplot3d import Axes3D


titles = ['Madame', 'Ms', 'Mansouer', 'Sir', 'Rev',
          'Mlle', 'Master', 'Col', 'Colonel', 'Major',
          'Mr', 'Captain', 'Dr', 'Lady', 'Miss', 'Mrs']


def read_training_data(filename):
    return pd.read_csv(filename)


def read_test_data(filename):
    data = pd.read_csv(filename, header=None)
    data.columns = ['Name', 'PClass', 'Age', 'Sex', 'Survived']
    return data


def map_str_to_ind(data, colname):
    unique_strs = data[colname].unique()
    unique_strs.sort()
    ids = dict()
    for i, unique_str in np.ndenumerate(unique_strs):
        ids[unique_str] = i
    return data[colname].replace(ids), ids


def map_title_to_ind(title):
    try:
        return titles.index(title) + 1
    except ValueError:
        return 0


def is_title(title):
    return title in titles


def fill_age(data):
    avg = int(data['Age'].mean())
    # print(data[math.isnan(data['Age'])])
    return data.fillna(avg)


def fill_missing(data):
    return fill_age(data)


def del_na_rows(data):
    return data.dropna(how='any', inplace=False)


def cols_with_nan(data):
    """ Test sample result: Age"""
    return list(filter(lambda x: data[x].isnull().any().any(), data.columns))


def reduce_age(age):
    if age < 6:
        return 0
    elif age < 18:
        return 1
    elif age < 40:
        return 2
    elif age < 50:
        return 3
    else:
        return 4


def clean(data, group_age=False):
    data = fill_missing(data)
    data['Name'], name_ids = map_str_to_ind(data, 'Name')
    # data['PClass'], pclass_ids = map_str_to_ind(data, 'PClass')
    data['PClass'] = data['PClass'].apply(map_pclass_to_ind)
    # data['Sex'], sex_id = map_str_to_ind(data, 'Sex')
    if group_age:
        data['Age'] = data['Age'].apply(reduce_age)
    del data['Sex']
    del data['Name']
    return data


def extract_title(name, gender=None):
    res = re.search(',\s\w+', name)
    if gender is None:
        if res is None:
            return name.split()[1]
        else:
            return res.group(0)[2:]
    cand = name.split()[1] if res is None else res.group(0)[2:]
    title = cand if is_title(cand) else ('Mr' if gender == 'male' else 'Mrs')
    return title


def map_pclass_to_ind(pclass):
    pclass = ord(pclass[:1]) - ord('0')
    return 1.5 if pclass < 0 else pclass


def title_to_gender(title):
    """
    :return: 0 if male, 1 if female
    :return type: int
    """
    female_titles = ['Ms', 'Mrs', 'Countess', 'Mme', 'Miss', 'Mlle', 'Lady']
    return int(title in female_titles)


def generate_features(data):
    data['Title'] = data.apply(lambda x: map_title_to_ind(extract_title(x['Name'], x['Sex'])), axis=1)
    return data


def preprocess(data, group_age=False):
    data = generate_features(data)
    data = clean(data, group_age)
    return data


def scale_features(features):
    # features = preprocessing.MinMaxScaler().fit_transform(features)
    features = preprocessing.scale(features)
    return features


def get_titles(data):
    return list(set(map(lambda x: extract_title(x), data['Name'])))


def extract(data):
    data['Survived'] = data['Survived'] + 1
    label = data['Survived'].as_matrix()
    del data['Survived']
    features = data.as_matrix()
    return features, label


def get_data(filename):
    data = read_training_data(filename)
    data = preprocess(data, group_age=True)
    return extract(data)


def get_test_data(filename):
    data = read_test_data(filename)
    data = preprocess(data, group_age=True)
    return extract(data)


def train_logreg(features, labels, cross_validate=False):
    if cross_validate:
        parameters = {'C': np.arange(0.1, 20, 0.5)}
        model = GridSearchCV(LogisticRegression(), parameters, n_jobs=8)
        return model.fit(features, labels)
    else:
        return LogisticRegression(C=10).fit(features, labels)


def train_decision_tree(features, labels):
    dtc = DecisionTreeClassifier(random_state=0)
    dtc.fit(features, labels)
    return dtc


def train_svm(features, labels, cross_validate=False):
    """
    :param features: Numpy feature matrix
    :param labels: Numpy feature vector
    :return: model
    """
    if cross_validate:
        parameters = {'kernel': ('linear', 'rbf'), 'C': np.arange(0.1, 20, 0.5)}
        model = GridSearchCV(SVC(probability=True), parameters, n_jobs=8)
        return model.fit(features, labels)
    else:
        model = SVC(probability=True)
        return model.fit(features, labels)


def train_bayes(features, labels):
    model = GaussianNB()
    model.fit(features, labels)
    return model


def run():
    features, labels = get_data('Data/train.csv')
    # generate_predictions(features, labels)
    plot_features(features, labels)


def generate_predictions(features, labels):
    print('Features:')
    print(features)

    f, truth = get_test_data('Data/test.csv')
    truth = truth - 1

    models = {'svm': train_svm(features, labels, cross_validate=True),
              'lgr': train_logreg(features, labels, cross_validate=True),
              'dtr': train_decision_tree(features, labels),
              'nbc': train_bayes(features, labels)}

    predictions = [model.predict(f) - 1 for key, model in models.items()]
    pred_prob = [model.predict_proba(f) for key, model in models.items()]

    print('SVM w/ Gaussian kernel: ', accuracy_score(truth, predictions[0]))
    print('LogReg accuracy: ', accuracy_score(truth, predictions[1]))
    print('Decision tree accuracy: ', accuracy_score(truth, predictions[2]))
    print('Naive Bayes accuracy: ', accuracy_score(truth, predictions[3]))

    print('Truth:              ', truth)
    print('SVM pred:           ', predictions[0])
    print('LogReg pred:        ', predictions[1])
    print('Decision tree pred: ', predictions[2])
    print('Naive bayes pred:   ', predictions[3])

    print('SVM prob pred:           ', pred_prob[0][:, 0])
    print('LogReg prob pred:        ', pred_prob[1][:, 0])
    print('Decision tree prob pred: ', pred_prob[2][:, 0])
    print('Naive bayes prob pred:   ', pred_prob[3][:, 0])

    return


def plot_features(features, labels):
    labels = labels - 1
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for f, l in zip(features, labels):
        if l:
            ax.scatter(f[0], f[1], f[2], color='r', marker='o')
        else:
            ax.scatter(f[0], f[1], f[2], color='b', marker='x')
    ax.set_xlabel('PClass')
    ax.set_ylabel('Age Group')
    ax.set_zlabel('Title')
    ax.set_zticks(np.arange(0, len(titles), 1))
    ax.set_zticklabels(titles)

    plt.show()


if __name__ == '__main__':
    run()

