# coding=utf-8
import math

import numpy as np
import pandas as pd
import pygraphviz as pgv
from TreeNode import TreeNode


# 选择最佳切分点
def chooseBestSplit(dataMat, hi, gi, r, T=0):
    # 获取特征数量
    featNum = dataMat.shape[1] - 1
    # 最大增益
    max_gain = float("-inf")
    # 切分特征下标
    split_feat = -1
    # 切分点的值
    split_value = -1
    # 切分后左子树的一阶导数和
    split_g_l = -1
    # 切分后右子树的一阶导数和
    split_g_r = -1
    # 切分后左子树的二阶导数和
    split_h_l = -1
    # 切分后右子树的二阶导数和
    split_h_r = -1
    # 遍历特征
    for feat in range(featNum):
        print '{}th feature'.format(feat)
        featList = dataMat[:, feat].T.tolist()[0]
        uniqueVals = sorted(set(featList))
        # 遍历特征值
        for value in uniqueVals:
            # 挑选比特征值小的样本，即左子树样本
            left_points = np.where(dataMat[:, feat] < value)
            # 挑选比特征值大的样本，即右子树样本
            right_points = np.where(dataMat[:, feat] >= value)
            # 左子树一阶导数和
            g_l = G(left_points, gi)
            # 右子树一阶导数和
            g_r = G(right_points, gi)
            # 左子树二阶导数和
            h_l = H(left_points, hi)
            # 右子树二阶导数和
            h_r = H(right_points, hi)
            # 计算分裂增益
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


# 进行分裂
def binSplitDataSet(dataMat, split_feat, split_value, hi, gi):
    left_points = np.where(dataMat[:, split_feat] < split_value)
    right_points = np.where(dataMat[:, split_feat] >= split_value)
    return dataMat[left_points[0]], dataMat[right_points[0]], hi[left_points[0]], hi[right_points[0]], gi[
        left_points[0]], gi[right_points[0]]


def createTree(dataMat, hi, gi, depth=0, max_depth=3, r=1, eta=0.1):
    # 选择最佳切分特征、最佳切分点、左右子树的一阶二阶导数和
    feat, val, g_l, g_r, h_l, h_r = chooseBestSplit(dataMat, hi, gi, r)
    root = TreeNode(feat, val)
    # 结点分裂，返回左右子结点每个样本的一阶导数和二阶导数值
    lSet, rSet, hi_l, hi_r, gi_l, gi_r = binSplitDataSet(dataMat, feat, val, hi, gi)
    # 如果数据集中样本个数大于1并且树的深度小于3层
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


# 计算叶子结点权重
def cal_weight(g, h, r):
    return -g / (h + r)


# 计算一阶导数和
def G(points, hi):
    return np.sum(hi[points])


# 计算二阶导数和
def H(points, gi):
    return np.sum(gi[points])


# 计算增益
def gain(g_l, h_l, g_r, h_r, r):
    left_gain = math.pow(g_l, 2) / (h_l + r)
    right_gain = math.pow(g_r, 2) / (h_r + r)
    all_gain = math.pow(g_l + g_r, 2) / (h_l + h_r + r)
    return left_gain + right_gain - all_gain


# 计算每个样本的一阶导数值
def g_i(y_pred, y_i):
    return y_pred - y_i


# 计算每个样本的二阶导数值
def h_i(y_pred):
    return y_pred * (1 - y_pred)


def load_data(path):
    data = pd.read_csv(path, dtype=np.float64, delimiter='\t', header=None)
    dataMat = np.mat(data.values)
    return dataMat


# 初始化一阶导数和二阶导数值
def init_base_score(trees, dataMat):
    label = dataMat[:, -1]
    if len(trees) == 0:
        base_score = np.zeros((dataMat.shape[0], 1))
        # 初始值设置为0.5，即base_score
        base_score += 0.5
        gi = g_i(base_score, label)
        hi = h_i(base_score)
    else:
        # 上一次预测值
        pred_res = predict(trees, dataMat)
        gi = g_i(pred_res, label)
        hi = h_i(pred_res)
    return hi, gi


# 预测函数
def predict(trees, dataMat):
    pred_res = np.zeros((dataMat.shape[0], 1), dtype=np.float64)
    for tree in trees:
        for i in range(dataMat.shape[0]):
            # 获取输入数据在每棵树上的输出
            weight = tree.get_weight(dataMat[i, :])
            # sigmoid变换
            pred_res[i, 0] += 1 / (1 + math.exp(-weight))
    return pred_res


# 画图
def draw_tree(root, i):
    A = pgv.AGraph(directed=True, strict=True)
    display(root, A)
    A.graph_attr['epsilon'] = '0.01'
    print A.string()  # print dot file to standard output
    A.write('tree_{}.dot'.format(i))
    A.layout('dot')  # layout with dot
    A.draw('tree_{}.png'.format(i))  # write to file


def display(root, A):
    if not root:
        return
    A.add_node(root.uid, label='x[{}]<{}'.format(root.split_feat, root.split_value))
    if root.left:
        if root.left.isLeaf:
            A.add_node(root.left.uid, label='leaf={}'.format(root.left.weight))
            A.add_edge(root.uid, root.left.uid, label='yes', color='red')
        else:
            A.add_node(root.left.uid, label='x[{}]<{}'.format(root.left.split_feat, root.left.split_value))
            A.add_edge(root.uid, root.left.uid, label='yes', color='red')
            display(root.left, A)

    if root.right:
        if root.right.isLeaf:
            A.add_node(root.right.uid, label='leaf={}'.format(root.right.weight))
            A.add_edge(root.uid, root.right.uid, label='no', color='blue')
        else:
            A.add_node(root.right.uid, label='x[{}]<{}'.format(root.right.split_feat, root.right.split_value))
            A.add_edge(root.uid, root.right.uid, label='no', color='blue')
            display(root.right, A)


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
    for i in range(len(trees)):
        print trees[i]
        draw_tree(trees[i], i)


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
