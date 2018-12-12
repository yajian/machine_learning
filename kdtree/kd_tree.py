# encoding:utf8
# 程序参考来源：https://huyesce.github.io/K-NN%20with%20Pyhton.html
import numpy as np
from node import Node
from collections import deque
import pygraphviz as pgv


def constructKDTree(dataset, datalabel):
    length = len(dataset)
    if length == 0:
        return
    median = length / 2
    # 计算维度个数
    dimension = len(dataset[0])
    # 求每个维度的方差，axis=0是按列运算的意思，axis=1是按行运算
    vars = np.var(dataset, axis=0)
    # 求方差最大的维度索引，即分割维度
    index = np.argmax(vars)
    # 按照方差最大的维度进行排序
    sorted_index = dataset[:, index].argsort()
    sorted_data = dataset[sorted_index]
    sorted_label = datalabel[sorted_index]
    # 左子树是小于中位数的部分
    left_data = sorted_data[:median]
    left_label = datalabel[:median]
    # 父节点是中位数
    median_data = sorted_data[median]
    median_label = datalabel[median]
    # 右子树是大于中位数的部分
    right_data = sorted_data[median + 1:]
    right_label = datalabel[median + 1:]
    # 递归建立kd树
    return Node(median_data, constructKDTree(left_data, left_label), constructKDTree(right_data, right_label), index,
                median_label)


def path_to_leaf_node(target, tree, path_deque):
    while tree:
        path_deque.append(tree)
        split = tree.split
        if target[split] < tree.point[split]:
            tree = tree.left
        else:
            tree = tree.right
    return path_deque


# 计算欧式距离函数
def distance(target, nearest_point):
    diff = nearest_point - target
    distance = np.sqrt(np.sum(diff ** 2))
    return distance


def find_nearest_neighbor(target, tree):
    # 记录到叶子结点节点前搜索过的节点
    path_deque = path_to_leaf_node(target, tree, deque())
    # 回溯第一个节点
    kd_node = path_deque.pop()
    # 假设第一个节点为最近邻
    nearest_point = kd_node.point
    # 计算最近邻节点与输入实例间距离
    nearest_dist = distance(target, nearest_point)
    # 最近邻节点的标签
    nearest_label = kd_node.label
    # 回溯
    while path_deque:
        kd_node = path_deque.pop()
        # 计算实例点与最近邻节点父节点的距离
        node_dist = distance(target, kd_node.point)
        # 更新最近邻节点
        if node_dist < nearest_dist:
            nearest_point = kd_node.point
            nearest_dist = node_dist
            nearest_label = kd_node.label
        # 获取分割维度
        s = kd_node.split
        # 判断是否需要进入最近邻节点的左右子空间进行搜索
        if abs(target[s] - kd_node.point[s]) < nearest_dist:
            # 进入右子树
            if target[s] < kd_node.point[s]:
                path_deque = path_to_leaf_node(target, kd_node.right, path_deque)
            # 进入左子树
            else:
                path_deque = path_to_leaf_node(target, kd_node.left, path_deque)

    return nearest_point, nearest_label, nearest_dist


def test_constructKDTree():
    dataset = np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
    datalebel = np.array([1, 1, 1, 0, 0, 0])
    target = [5.1, 3.1]
    kdtree = constructKDTree(dataset, datalebel)
    A = pgv.AGraph(directed=True, strict=True)
    display(kdtree, A)
    A.graph_attr['epsilon'] = '0.01'
    print A.string()  # print dot file to standard output
    A.write('tree.dot')
    A.layout('dot')  # layout with dot
    A.draw('tree.png')  # write to file


def display(kdtree, A):
    if kdtree.left != None:
        A.add_edge(kdtree.point, kdtree.left.point)
        display(kdtree.left, A)
    if kdtree.right != None:
        A.add_edge(kdtree.point, kdtree.right.point)
        display(kdtree.right, A)


def main():
    test_constructKDTree()


# dataset = np.array([[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]])
# target=[5.1,3.1]
# datalabel = np.array([1,1,1,0,0,0])
# kdtree = constructKDTree(dataset,datalabel)
# nearest_point,nearest_label,nearest_dist = find_nearest_neighbor(target,kdtree)
# print nearest_point
# print nearest_label
# print nearest_dist


if __name__ == "__main__":
    main()
