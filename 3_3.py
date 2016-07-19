# -*- coding: cp936 -*-
from numpy import *
import pandas as pd
import matplotlib.pyplot as plt

#读入csv文件数据
df=pd.read_csv('3.csv')
m,n=shape(df.values)
df['norm']=ones((m,1))
dataMat=array(df[['norm','density','ratio_sugar']].values[:,:])
labelMat=mat(df['label'].values[:]).transpose()
positiveMat=mat(df['good'].values[:]).transpose()
#sigmoid函数
def sigmoid(inX):
    return 1.0/(1+exp(-inX))

#梯度上升算法
def gradAscent(dataMat,labelMat):
    m,n=shape(dataMat)
    alpha=0.1
    maxCycles=5000
    weights=array(ones((n,1)))

    for k in range(maxCycles):
        a=dot(dataMat,weights)
        h=sigmoid(a)
        error=(labelMat-h)
        weights=weights+alpha*dot(dataMat.transpose(),error)
    return weights


#画图
def plotBestFit(weights):
    m=shape(dataMat)[0]
    xcord1=[]
    ycord1=[]
    xcord2=[]
    ycord2=[]
    for i in range(m):
        if positiveMat[i]==1:
            xcord1.append(dataMat[i,1])
            ycord1.append(dataMat[i,2])
        else:
            xcord2.append(dataMat[i,1])
            ycord2.append(dataMat[i,2])
    density = []
    for i in dataMat:
        density.append(i[1])
    plt.figure(1)
    ax=plt.subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x=[min(density),max(density)]
    y=array((-(weights[0]+weights[1]*x))/weights[2])

    plt.sca(ax)
    plt.plot(x,y[0])   #gradAscent
    plt.xlabel('density')
    plt.ylabel('ratio_sugar')
    plt.title('gradAscent logistic regression')
    plt.show()

weights=gradAscent(dataMat,positiveMat)
plotBestFit(weights)