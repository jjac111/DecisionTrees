import pandas as pd
import numpy as np
from random import choice
from anytree import Node, RenderTree
from statistics import mode, StatisticsError
from math import log2


class Tree:

    def __init__(self, train_data, type='CART', branches_per_split=10):
        self.train_data = train_data
        self.type = type
        self.root = Node('root', data=train_data, features=list(train_data.columns)[1:])
        self.branches_per_split = branches_per_split

    def fit(self):
        nodes_to_expand = [self.root]

        while len(nodes_to_expand) != 0:
            node = nodes_to_expand.pop()
            data = node.data
            y_col = data.columns[0]
            targets = np.unique(data[y_col])

            if len(targets) == 1:
                node.label = targets[0]
                node.name = ' => '.join([node.name, str(int(node.label))])
                continue
            elif len(node.features) == 0:
                try:
                    # suspicious
                    label = mode(node.parent.data[y_col])
                except StatisticsError as e:
                    label = choice(np.unique(node.parent.data[y_col]))
                node.name = ' => '.join([node.name, str(int(node.label))])
                continue
            elif len(node.data) == 0:
                try:
                    # suspicious
                    label = mode(node.parent.data[y_col])
                except StatisticsError as e:
                    label = choice(np.unique(node.parent.data[y_col]))
                node.label = label
                node.name = ' => '.join([node.name, str(int(node.label)) + '(uncertain)'])
                continue

            ft = self.select_feature(node.data, y_col, node.features, targets)

            ft_col = data[ft]
            bins = np.linspace(min(ft_col), max(ft_col), self.branches_per_split + 1)
            bins = [(bins[i], bins[i + 1]) for i in range(len(bins) - 1)]

            for bin in bins:
                new_node_data = data[(data[ft] >= bin[0]) & (data[ft] < bin[1])]

                if bin[1] == max(ft_col):
                    new_node_data.append(data[data[ft] == max(ft_col)])

                if new_node_data.empty:
                    continue

                new_features = node.features.copy()
                new_features.remove(ft)
                condition = lambda x: x >= bin[0] & x < bin[1]
                new_node = Node(f'F{ft}', parent=node, data=new_node_data, features=new_features, condition=condition)

                nodes_to_expand.append(new_node)

        print('The fit tree looks like this:')
        for pre, _, node in RenderTree(self.root):
            print("%s%s" % (pre, node.name))

    def predict(self):
        pass

    def select_feature(self, curr, y_col, features, targets):

        probs = [len(curr[curr[y_col] == t]) / len(curr) for t in targets]
        info = - sum([p * log2(p) for p in probs])
        gini = 1 - sum([p ** 2 for p in probs])
        measures = []

        for ft in features:
            ft_col = curr[ft]
            bins = np.linspace(min(ft_col), max(ft_col), self.branches_per_split + 1)
            bins = [(bins[i], bins[i + 1]) for i in range(len(bins) - 1)]
            binned = [curr[(curr[ft] >= bin[0]) & (curr[ft] < bin[1])] for bin in bins]
            binned[-1].append(curr[curr[ft] == max(ft_col)])
            binned = [df for df in binned if not df.empty]

            if self.type == 'ID3' or self.type == 'C4.5':
                info_ft = -sum([(len(df) / (len(curr))) * (
                        1 - sum([p * log2(p) for p in [len(df[df[y_col] == t]) / len(df) for t in targets]])) for df in
                                binned])
                gain = info - info_ft

                if self.type == 'ID3':
                    to_append = (ft, gain)
                elif self.type == 'C4.5':
                    split_info = -sum([(len(df) / len(curr)) * log2(len(df) / len(curr)) for df in binned])
                    to_append = (ft, gain / split_info)

            elif self.type == 'CART':
                gini_ft = sum([(len(df) / (len(curr))) * (
                        1 - sum([p ** 2 for p in [len(df[df[y_col] == t]) / len(df) for t in targets]])) for df in
                               binned])
                impurity = gini - gini_ft
                to_append = (ft, impurity)

            measures.append(to_append)

        return min(measures, key=lambda x: x[1])[0]
