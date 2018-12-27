# coding=utf-8
from numpy import *
import os


def createVocabList(dataSet):
    vocabSet = set()
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


# 制作词袋模型
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print 'the word {} is not in my Vocabulary!'.format(word)
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    # 包含敏感词的此条出现的概率
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # 构造单词出现次数列表 [1,1,...,1],这里使用了拉普拉斯平滑的思想
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        # 是否是侮辱性文件
        if trainCategory[i] == 1:
            # 对侮辱性文件的向量进行加和,[0,1,1,....] + [0,1,1,....]->[0,2,2,...]
            p1Num += trainMatrix[i]
            # 计算所有侮辱性文件中出现的单词总数
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 在1类别下，每个单词出现的概率,[1,2,3,5]/90->[1/90,...]
    p1Vect = log(p1Num / p1Denom)
    #  在0类别下，每个单词出现的概率
    p0Vect = log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    # p1Vec是每个词在类别1里出现的概率，vec2Classify表示这个词在该文章中是否出现过，乘起来表示文章中每个词出现的概率
    # 这里进行了log变换，所以写成了加法
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testParse(bigString):
    import re
    listOfTokens = re.split('r\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        # 读取非垃圾邮件文本
        wordList = testParse(open(os.getcwd() + '/email/spam/{}.txt'.format(i)).read())
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(1)
        # 读取垃圾邮件文本
        wordList = testParse(open(os.getcwd() + '/email/ham/{}.txt'.format(i)).read())
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(0)
    # 建立词集
    vocabList = createVocabList(docList)
    # 训练集语料的下标
    trainingSet = range(50)
    testSet = []
    # 选10个语料作为测试集
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        # 删除后训练集变成40个
        del (trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    # 计算各种概率
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    # 进行测试
    for docIndex in testSet:
        # 构建词向量
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        # 进行分类
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print 'the error rate is {}'.format(float(errorCount) / len(testSet))


def main():
    spamTest()


if __name__ == '__main__':
    main()
