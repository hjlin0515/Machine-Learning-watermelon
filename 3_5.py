# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('3.csv')


def calulate_w():
    df0 = df[df.good==0]
    df1 = df[df.good==1]
    x0 = df0.values[:,1:3]
    x1 = df1.values[:,1:3]
    mean0 = np.array([np.mean(x0[:,0]), np.mean(x0[:,1])])
    mean1 = np.array([np.mean(x1[:,0]), np.mean(x1[:,1])])
    m0 = np.shape(x0)[0]
    Sw = np.zeros(shape=(2,2))

    for i in range(m0):
        temp = np.mat(x0[i,:]-mean0)
        Sw = Sw + temp.transpose() * temp

    m1 = np.shape(x1)[0]
    for i in range(m1):
        temp = np.mat(x1[i,:]-mean1)
        Sw = Sw + temp.transpose() * temp

    w = (mean0-mean1)*(np.mat(Sw).I)
    return w

def plot(w):
    dataMat=np.array(df[['density','ratio_sugar']].values[:,:])
    labelMat=np.mat(df['good'].values[:]).transpose()
    m=np.shape(dataMat)[0]
    xcord1=[]
    ycord1=[]
    xcord2=[]
    ycord2=[]
    for i in range(m):
        if labelMat[i]==1:
            xcord1.append(dataMat[i,0])
            ycord1.append(dataMat[i,1])
        else:
            xcord2.append(dataMat[i,0])
            ycord2.append(dataMat[i,1])
    plt.figure(1)
    ax=plt.subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x=np.arange(-0.2,0.8,0.1)
    y=np.array((-w[0,0]*x)/w[0,1])

    plt.sca(ax)
    plt.plot(x,y)   #gradAscent
    plt.xlabel('density')
    plt.ylabel('ratio_sugar')
    plt.title('LDA')
    plt.show()

plot(calulate_w())