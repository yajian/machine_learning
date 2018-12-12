# coding=utf-8
from math import log
import pygraphviz as pgv


def calcShannonEnt(dataSet):
    # 数据集总条数
    numEntries = len(dataSet)
    labelCount = {}
    # 计算每个类别样本的数量
    for featVec in dataSet:
        currentLabel = featVec[-1]
        labelCount[currentLabel] = labelCount.get(currentLabel, 0) + 1
    shannonEnt = 0.0
    for key in labelCount:
        # 计算每个类别出现的概率
        prob = float(labelCount[key]) / numEntries
        # 计算信息熵
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        # 挑选第axis个特征的值为value的数据
        if featVec[axis] == value:
            reducedFeature = featVec[:axis]
            reducedFeature.append(featVec[axis + 1:])
            # 注意，此处是extend，不是append
            retDataSet.extend(reducedFeature)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    # 特征数量，由于最后一个是标签，所以减1
    numFeatures = len(dataSet[0]) - 1
    # 计算经验熵
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        # 把第i个特征全部挑出
        featList = [example[i] for example in dataSet]
        # 第i个特征有几个不同的特征值
        uniqueVals = set(featList)
        # 条件经验熵
        newEntropy = 0.0
        for value in uniqueVals:
            # 把特征值等于value的数据挑出
            subDataSet = splitDataSet(dataSet, i, value)
            # 计算特征值等于value的数据占总样本的比例
            prob = len(subDataSet) / float(len(dataSet))
            # 计算经验条件熵
            newEntropy += prob * calcShannonEnt(subDataSet)
        # 计算信息增益
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        classCount[vote] = classCount.get(vote, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=lambda item: item[1], reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


def createDataSet():
    dataSet = [['1', '1', 'yes'],
               ['1', '1', 'yes'],
               ['1', '0', 'no'],
               ['0', '1', 'no'],
               ['0', '1', 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def drawTree(tree, A):
    for key, value in tree.items():
        if isinstance(value, dict):
            for kk, vv in value.items():
                if isinstance(vv, dict):
                    for kkk, vvv in vv.items():
                        A.add_edge(key, kkk, label=kk)
                    drawTree(vv, A)
                else:
                    A.add_edge(key, vv, label=kk)


def display(tree):
    A = pgv.AGraph(directed=True, strict=True)
    drawTree(tree, A)
    A.graph_attr['epsilon'] = '0.01'
    print A.string()  # print dot file to standard output
    A.write('tree.dot')
    A.layout('dot')  # layout with dot
    A.draw('tree.png')  # write to file


def main():
    dataSet, labels = createDataSet()
    tree = createTree(dataSet, labels)
    print tree
    display(tree)


if __name__ == '__main__':
    main()
