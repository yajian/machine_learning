# coding=utf-8
import math

import numpy as np
import pandas as pd

from TreeNode import TreeNode


def chooseBestSplit(dataMat, hi, gi, r, T=0):
    featNum = dataMat.shape[1] - 1
    max_gain = float("-inf")
    split_feat = -1
    split_value = -1
    split_g_l = -1
    split_g_r = -1
    split_h_l = -1
    split_h_r = -1
    for feat in range(featNum):
        print '{}th feature'.format(feat)
        featList = dataMat[:, feat].T.tolist()[0]
        uniqueVals = sorted(set(featList))
        for value in uniqueVals:
            left_points = np.where(dataMat[:, feat] < value)
            right_points = np.where(dataMat[:, feat] >= value)
            g_l = G(left_points, gi)
            g_r = G(right_points, gi)
            h_l = H(left_points, hi)
            h_r = H(right_points, hi)
            g = gain(g_l, h_l, g_r, h_r, r)
            print '{}-g_l:{}, h_l:{}, g_r:{}, h_r:{}-g:{}'.format(value, g_l, h_l, g_r, h_r, g)
            if g >= max_gain:
                max_gain = g
                split_feat = feat
                split_value = value
                split_g_l = g_l
                split_g_r = g_r
                split_h_l = h_l
                split_h_r = h_r
    return split_feat, split_value, split_g_l, split_g_r, split_h_l, split_h_r


def binSplitDataSet(dataMat, split_feat, split_value, hi, gi):
    left_points = np.where(dataMat[:, split_feat] < split_value)
    right_points = np.where(dataMat[:, split_feat] >= split_value)
    return dataMat[left_points[0]], dataMat[right_points[0]], hi[left_points[0]], hi[right_points[0]], gi[
        left_points[0]], gi[right_points[0]]


def createTree(dataMat, hi, gi, depth=0, max_depth=3, r=1, eta=0.1):
    feat, val, g_l, g_r, h_l, h_r = chooseBestSplit(dataMat, hi, gi, r)
    root = TreeNode(feat, val)
    lSet, rSet, hi_l, hi_r, gi_l, gi_r = binSplitDataSet(dataMat, feat, val, hi, gi)
    if len(set(lSet[:, -1].T.tolist()[0])) > 1 and depth + 1 < max_depth:
        root.left = createTree(lSet, hi_l, gi_l, depth + 1)
    else:
        leaf = TreeNode(-1, -1)
        leaf.weight = eta * cal_weight(g_l, h_l, r)
        leaf.isLeaf = True
        root.left = leaf

    if len(set(rSet[:, -1].T.tolist()[0])) > 1 and depth + 1 < max_depth:
        root.right = createTree(rSet, hi_r, gi_r, depth + 1)
    else:
        leaf = TreeNode(-1, -1)
        leaf.weight = eta * cal_weight(g_r, h_r, r)
        leaf.isLeaf = True
        root.right = leaf
    return root


def cal_weight(g, h, r):
    return -g / (h + r)


def G(points, hi):
    return np.sum(hi[points])


def H(points, gi):
    return np.sum(gi[points])


def gain(g_l, h_l, g_r, h_r, r):
    left_gain = math.pow(g_l, 2) / (h_l + r)
    right_gain = math.pow(g_r, 2) / (h_r + r)
    all_gain = math.pow(g_l + g_r, 2) / (h_l + h_r + r)
    return left_gain + right_gain - all_gain


def g_i(y_pred, y_i):
    return y_pred - y_i


def h_i(y_pred):
    return y_pred * (1 - y_pred)


def load_data(path):
    data = pd.read_csv(path, dtype=np.float64, delimiter='\t', header=None)
    dataMat = np.mat(data.values)
    return dataMat


def init_base_score(trees, dataMat):
    label = dataMat[:, -1]
    if len(trees) == 0:
        base_score = np.zeros((dataMat.shape[0], 1))
        base_score += 0.5
        gi = g_i(base_score, label)
        hi = h_i(base_score)
    else:
        pred_res = predict(trees, dataMat)
        gi = g_i(pred_res, label)
        hi = h_i(pred_res)
    return hi, gi


def predict(trees, dataMat):
    pred_res = np.zeros((dataMat.shape[0], 1), dtype=np.float64)
    for tree in trees:
        for i in range(dataMat.shape[0]):
            weight = tree.get_weight(dataMat[i, :])
            pred_res[i, 0] += 1 / (1 + math.exp(-weight))
    return pred_res


def main():
    dataMat = load_data('./data.txt')
    root = None
    trees = []
    tree_num = 2
    for i in range(tree_num):
        print '{}th tree building'.format(i)
        hi, gi = init_base_score(trees, dataMat)
        root = createTree(dataMat, hi, gi)
        trees.append(root)
    for tree in trees:
        print tree


if __name__ == '__main__':
    main()

# def xgboost(dataMat):
#     ##直接使用xgboost
#     dtrain = xgb.DMatrix(dataMat[:, 0:2], label=dataMat[:, -1])
#
#     param = {
#         'booster': 'gbtree',
#         'objective': 'reg:logistic',
#         'eval_metric': 'logloss',
#         'max_depth': 3,
#         'eta': 0.1,
#         'lambda': 1,
#         'min_child_weight': 0,
#         'gamma': 0,
#         'base_score': 0.5,
#         'silent': 0,
#         'n_estimators': 2
#     }
#     bst = xgb.train(param, dtrain, 2)
#
#     from matplotlib import pyplot
#
#     xgb.plot_tree(bst, num_trees=0)
#     pyplot.show()
