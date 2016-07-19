# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from math import log

import operator


#对离散变量进行划分数据集
def splitDataSet(dataSet,axis,value):
    resDataSet = []
    for vec in dataSet:
        if vec[axis] == value:
            reducedVec = vec[:axis]
            reducedVec.extend(vec[axis + 1:])
            resDataSet.append(reducedVec)
    return resDataSet

#计算信息熵
def getEnt(dataSet):
    num = len(dataSet)
    labelsCount = {}
    for vec in dataSet:
        currentLabel = vec[-1]
        if currentLabel not in labelsCount.keys():
            labelsCount[currentLabel] = 1
        else:
            labelsCount[currentLabel] += 1
    Ent = 0.0
    for key in labelsCount:
        prob = float(labelsCount[key])/num
        Ent -= prob*log(prob,2)
    return Ent

#对连续变量进行划分数据集 opt表示应该大于(1)还是小于(0)
def splitContinuousDataSet(dataSet,axis,value,opt):
    resDataSet = []
    for vec in dataSet:
        if opt == 0:
            if vec[axis] <= value:
                reducedVec = vec[:axis]
                reducedVec.extend(vec[axis+1:])
                resDataSet.append(reducedVec)
        else:
            if vec[axis] > value:
                reducedVec = vec[:axis]
                reducedVec.extend(vec[axis + 1:])
                resDataSet.append(reducedVec)
    return resDataSet

#选择最优划分属性
def chooseBestFeatureToSplit(dataSet,labels):
    numFeature = len(dataSet[0])-1
    baseEnt = getEnt(dataSet)
    bestSplitDict = {}
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeature):
        featList = [example[i] for example in dataSet]
        #对连续型特征
        if type(featList[0]).__name__ == 'float' or type(featList[0]).__name__ == 'int':
            sortFeatList = sorted(featList)
            splitList = []
            #产生n-1个划分点
            for j in range(len(sortFeatList)-1):
                splitList.append((sortFeatList[j]+sortFeatList[j+1])/2.0)
            bestSplitEnt = 100000
            for j in range(len(splitList)):
                value = splitList[j]
                newEnt = 0.0
                subDataSet0 = splitContinuousDataSet(dataSet,i,value,0)
                subDataSet1 = splitContinuousDataSet(dataSet,i,value,1)
                prob0 = len(subDataSet0)/float(len(dataSet))
                newEnt += prob0*getEnt(subDataSet0)
                prob1 = len(subDataSet1)/float(len(dataSet))
                newEnt += prob1*getEnt(subDataSet1)
                if newEnt < bestSplitEnt:
                    bestSplitEnt = newEnt
                    bestSplit = j
            bestSplitDict[labels[i]] = splitList[bestSplit]
            infoGain = baseEnt - bestSplitEnt
        #对离散型特征
        else:
            uniqueVals = set(featList)
            newEnt = 0.0
            for value in uniqueVals:
                subDataSet = splitDataSet(dataSet,i,value)
                prob = len(subDataSet)/float(len(dataSet))
                newEnt += prob*getEnt(subDataSet)
            infoGain = baseEnt - newEnt
        if infoGain>bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    if type(dataSet[0][bestFeature]).__name__ == 'float' or type(dataSet[0][bestFeature]).__name__ == 'int':
        bestSplitValue = bestSplitDict[labels[bestFeature]]
        labels[bestFeature] = labels[bestFeature] + "<=" + str(bestSplitValue)
        for i in range(np.shape(dataSet)[0]):
            if dataSet[i][bestFeature] <= bestSplitValue:#以连续特征进行划分时，将记录进行二值化处理
                dataSet[i][bestFeature] = 1
            else:
                dataSet[i][bestFeature]=0
    return bestFeature


#获取所含样本最多的类
def getMajority(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 1
        else:
            classCount[vote] += 1
    return max(classCount)


# 生成决策树
def createTree(dataSet, labels, data_full, labels_full):
    classList = [i[-1] for i in dataSet]
    #当前结点包含的样本全属于同一类，无须再分
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    #当前结点属性集为空，无法分类
    if len(dataSet[0]) == 1:
        return getMajority(classList)
    bestFeature = chooseBestFeatureToSplit(dataSet,labels)
    bestFeatureLabel = labels[bestFeature]
    myTree={bestFeatureLabel:{}}
    featureVals = [i[bestFeature] for i in dataSet]
    uniqueVals = set(featureVals)
    if type(dataSet[0][bestFeature]).__name__ == 'str':
        currentLabel = labels_full.index(labels[bestFeature])
        featureValsFull = [i[currentLabel] for i in data_full]
        uniqueValsFull = set(featureValsFull)
    del(labels[bestFeature])
    for value in uniqueVals:
        subLabels = labels[:]
        if type(dataSet[0][bestFeature]).__name__ == 'str':
            uniqueValsFull.remove(value)
        myTree[bestFeatureLabel][value] = createTree(splitDataSet(dataSet,bestFeature,value),subLabels,data_full,labels_full)
    if type(dataSet[0][bestFeature]).__name__ == 'str':
        for value in uniqueValsFull:
            myTree[bestFeatureLabel][value] = getMajority(classList)
    return myTree




df = pd.read_csv('watermelon_3.0.csv')
data = df.values[:, 1:].tolist()
data_full = data[:]
labels = df.columns.values[1:-1].tolist()
labels_full = labels[:]
DecisionTree = createTree(data, labels, data_full, labels_full)

