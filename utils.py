import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, \
    precision_recall_curve
from statistics import mean, stdev
from math import floor
from config import *
from decision_tree import *
from IPython.display import display
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt


def prepare_data():
    if not os.path.exists(preprocessed_data_path):
        os.mkdir(preprocessed_data_path)

    raw_train = pd.read_csv(train_filename, header=None)
    raw_test = pd.read_csv(test_filename, header=None)

    merged = pd.concat([raw_train, raw_test], axis=0, ignore_index=True).drop(columns=features_to_remove).sample(frac=1)

    scaler = MinMaxScaler()

    print('Raw dataset:')
    display(merged)

    merged = pd.DataFrame(scaler.fit(merged).transform(merged))

    print('Normalized dataset:')
    display(merged)

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


def classify(type='None', plot_all_trees=False):
    metrics = {}
    plotted_once = False
    for i in range(10):
        metrics[i] = {}

        test = pd.read_csv('/'.join([ten_fold_data_path, f'{i + 1}_test.csv']))
        train = pd.read_csv('/'.join([ten_fold_data_path, f'{i + 1}_train.csv']))

        classifier = Tree(train, type, branches_per_split=5)

        classifier.fit()

        if plot_all_trees:
            classifier.plot_tree()
        elif not plotted_once:
            classifier.plot_tree()
            plotted_once = True

        predicted = classifier.predict(test)

        test_y = test[test.columns[0]]
        metrics['type'] = type
        metrics[i]['confusion_matrix'] = pd.DataFrame(confusion_matrix(test_y, predicted),
                                                      index=['Actual Neg', 'Actual Pos'],
                                                      columns=['Predicted Neg', 'Predicted Pos'])
        metrics[i]['accuracy'] = accuracy_score(test_y, predicted)
        metrics[i]['precision'] = precision_score(test_y, predicted)
        metrics[i]['recall'] = recall_score(test_y, predicted)
        metrics[i]['roc_auc'] = roc_auc_score(test_y, predicted)
        metrics[i]['roc_curve'] = (roc_curve(test_y, predicted))
        metrics[i]['precision_recall_curve'] = (precision_recall_curve(test_y, predicted))
        print(i + 1, end=' ')


    return metrics


def plot_results(metrics):
    type = metrics.pop('type')
    confusion_matrices = [v['confusion_matrix'] for k, v in metrics.items()]
    accuracies = [v['accuracy'] for k, v in metrics.items()]
    precisions = [v['precision'] for k, v in metrics.items()]
    recalls = [v['recall'] for k, v in metrics.items()]
    aucs = [v['roc_auc'] for k, v in metrics.items()]

    print('\n_______________________________________________________________________________')
    print(f'{type} Tree Results')
    print('Confusion Matrices:')
    for i, m in enumerate(confusion_matrices):
        print(f'\n{i+1} Fold:')
        display(m)

    print('Accuracy mean:', mean(accuracies), sep='\n')
    print('Accuracy stdev:', stdev(accuracies), '\n', sep='\n')

    print('Precision mean:', mean(precisions), sep='\n')
    print('Precision stdev:', stdev(precisions), '\n', sep='\n')

    print('Recalls mean:', mean(recalls), sep='\n')
    print('Recalls stdev:', stdev(recalls), '\n', sep='\n')

    print('AUC mean:', mean(aucs), sep='\n')
    print('AUC stdev:', stdev(aucs), '\n', sep='\n')

    num_plots = len(metrics.keys())
    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(nrows=2, ncols=5)
    for i in range(num_plots):
        row, col = floor(i / 5), i % 5
        ax = fig.add_subplot(gs[row, col])
        ax.plot(metrics[i]['roc_curve'][0], metrics[i]['roc_curve'][1])
        ax.set_title(f'ROC Curve - {i + 1}th Fold')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
    plt.show()

    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(nrows=2, ncols=5)
    for i in range(num_plots):
        row, col = floor(i / 5), i % 5
        ax = fig.add_subplot(gs[row, col])
        ax.plot(metrics[i]['precision_recall_curve'][0], metrics[i]['precision_recall_curve'][1])
        ax.set_title(f'Precision vs. Recall - {i + 1}th Fold')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
    plt.show()
