import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, auc
from statistics import mean, stdev
from config import *
from decision_tree import *


def prepare_data():
    if not os.path.exists(preprocessed_data_path):
        os.mkdir(preprocessed_data_path)

    raw_train = pd.read_csv(train_filename, header=None)
    raw_test = pd.read_csv(test_filename, header=None)

    merged = pd.concat([raw_train, raw_test], axis=0, ignore_index=True).drop(columns=features_to_remove).sample(frac=1)

    scaler = MinMaxScaler()

    print('Raw dataset:')
    print(merged)

    merged = pd.DataFrame(scaler.fit(merged).transform(merged))

    print('Normalized dataset:')
    print(merged)

    merged.to_csv(preprocessed_data_path + 'normalized.csv', header=False, index=False)


def make_folded_sets():
    if not os.path.exists(ten_fold_data_path):
        os.mkdir(ten_fold_data_path)

    merged = pd.read_csv(preprocessed_data_path + 'normalized.csv', header=None)

    length = int(len(merged) / 10)  # length of each fold
    folds = []
    for i in range(9):
        folds += [merged.iloc[i * length:(i + 1) * length]]
    folds += [merged.iloc[9 * length:len(merged)]]

    for i, test in enumerate(folds):
        train = merged.drop(index=test.index)

        test.to_csv('/'.join([ten_fold_data_path, f'{i + 1}_test.csv']), index=None)
        train.to_csv('/'.join([ten_fold_data_path, f'{i + 1}_train.csv']), index=None)


def classify(type='None'):
    metrics = {}
    for i in range(10):
        metrics[i] = {}

        test = pd.read_csv('/'.join([ten_fold_data_path, f'{i + 1}_test.csv']))
        train = pd.read_csv('/'.join([ten_fold_data_path, f'{i + 1}_train.csv']))

        classifier = Tree(train, type, branches_per_split=5)

        classifier.fit()

        predicted = classifier.predict(test)

        test_y = test[test.columns[0]]
        metrics[i]['accuracy'] = accuracy_score(test_y, predicted)
        metrics[i]['precision'] = precision_score(test_y, predicted)
        metrics[i]['recall'] = recall_score(test_y, predicted)
        metrics[i]['auc'] = auc(test_y, predicted)

    return metrics