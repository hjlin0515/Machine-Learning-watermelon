# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

def sigmod(x):
    return 1./(1+np.exp(-x))




df = pd.read_csv('watermelon_3.0.csv')[['density', 'sugar_ratio', 'label']]
df = df.iloc[np.argsort(np.random.randn(df.shape[0]))]
np.random.seed(20160618)

X = df.values[:, :-1]
y = df.values[:, -1]
y = y.reshape(y.shape[0], 1)


max_loop = 10
c_hidden = 10


#随机初始化参数
hidden_weight = np.array([np.random.randn(c_hidden) for i in range(X.shape[1])])
output_weight = np.array([np.random.randn(y.shape[1]) for i in range(c_hidden)])
hidden_threshold = np.random.randn(c_hidden).reshape(c_hidden, 1)
output_threshold = np.random.randn(y.shape[1]).reshape(y.shape[1], 1)



#学习率
eta = [7e-4, 3e-4]
for step in range(max_loop):
    for sample in range(X.shape[0]):
        x = X[sample].reshape(1,X.shape[1])
        #hidden的输出
        hh = np.array([sigmod(v - hidden_threshold[i])
                      for i,v in enumerate(np.dot(x,hidden_weight))])
        #ouput的输出
        yy = np.array([sigmod(v - output_threshold[i])
                       for i,v in enumerate(np.dot(hh,output_weight))]).reshape(y.shape[1],1)
        g = (y[sample]-yy) * yy * (1 - yy)
        e = np.dot(output_weight,g)*(hh*(1-hh)).reshape(hh.shape[1],-1)
        #更新参数
        output_weight += (eta[1]*g*hh).reshape(output_weight.shape[0],-1)
        output_threshold += (-1*eta[1]*g).reshape(output_threshold.shape[0],-1)
        hidden_weight += (eta[0]*e*x).reshape(hidden_weight.shape[0],-1)
        hidden_threshold += (-1*eta[0]*e).reshape(hidden_threshold.shape[0],-1)


def predict_x(x):
    x = x.reshape(1, x.shape[0])
    hh = np.array([sigmod(v - hidden_threshold[i])
                   for i, v in enumerate(np.dot(x, hidden_weight))])
    yy = np.array([sigmod(v - output_threshold[i])
                   for i, v in enumerate(np.dot(hh, output_weight))])
    return 1 if yy[0] > .5 else 0
def predict(X):
    return np.array([predict_x(x) for x in X])

def score(X,y):
    return (predict(X) == y.reshape(1,y.shape[0])).sum() * 1. / X.shape[0]

y_predict = predict(X)
s = score(X,y)
print s
print roc_auc_score(y,predict(X))#AUC