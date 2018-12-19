# coding=utf-8

from numpy import *


# 读取数据
def loadDataSet(path):
    dataMat = []
    fr = open(path, 'rb')
    for line in fr.readlines():
        currLine = line.strip().split('\t')
        fltLine = map(float, currLine)
        dataMat.append(fltLine)
    return mat(dataMat)


# 以value为分界点切分数据集
def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1


# 计算叶子结点中数据的均值
def regLeaf(dataSet):
    return mean(dataSet[:, -1])


# 计算误差，即数据集的平方误差，这里使用方差乘以总个数计算
def regErr(dataSet):
    return var(dataSet[:, -1]) * shape(dataSet)[0]


# 二元切分选择分裂点
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    tolS = ops[0]
    tolN = ops[1]
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    m, n = shape(dataSet)
    S = errType(dataSet)
    bestS = inf
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n - 1):
        for splitVal in set(dataSet[:, featIndex].T.tolist()[0]):
            # 切分数据集
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            # 判断切分后的数据集的条数是否满足要求
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
                continue
            # 计算两个数据集的误差
            newS = errType(mat0) + errType(mat1)
            # 选取误差最小的切分方式
            if newS < S:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    # 如果误差已经小于要求的误差
    if S - bestS < tolS:
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    # 如果不满足条数要求，不再继续分裂，返回结点的值
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    return bestIndex, bestValue


def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    # 选择最佳分裂点
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None:
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    # 左子树
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    # 柚子树
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree


def main():
    data = loadDataSet('./data.txt')
    tree = createTree(data)
    print tree


if __name__ == '__main__':
    main()
