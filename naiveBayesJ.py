#!/usr/bin/python3
# coding=utf-8

import numpy as np
import pandas as pd
import math
from sklearn.base import BaseEstimator
class NaiveBayesJ(BaseEstimator):
    #_BayesianEstimation 含拉普拉斯平滑
    # cat_cols传入离散类别特诊的列号
    def __init__(self):
        self.means=[]
        self.vars=[]
        self.feature_pct=[]
        self.cat_cols=[]
        self.num_cols=[]
        self.target_unique=[]
        self.c_pct=[]
        pass
    def fit(self,data,target,cat_cols=[]):
        # class_num=len(np.unique(target)) # 种类总数
        self.target_unique=np.unique(target)
        data_rows=data.shape[0]
        for c in self.target_unique:# 对每种class
            c_count=sum(target==c)
            self.c_pct.append(c_count/data_rows)
            d_c=data[target==c,:]
            if len(cat_cols)>0:# 有离散类别特征
                self.cat_cols=cat_cols
                data_cat=d_c[:,cat_cols]
                feature_pct_onec=[]
                for cat_f in cat_cols:# 对每种类别feature
                    Sj=len(np.unique(data_cat[:cat_f]))# 特征cat_f的不同取值数
                    dic_tmp={}# 临时存储1个feature、各种取值的占比
                    for one_cat_f_value in  np.unique(data_cat[:cat_f]):# 对每种类别特征的1个取值
                        dic_tmp[one_cat_f_value]=(sum(data_cat[:cat_f]==one_cat_f_value)+1)/(c_count+Sj)# 类别为c，特征cat取值为one_cat_f_value的占比
                    self.feature_pct_onec.append(dic_tmp)
                self.feature_pct.append(feature_pct_onec)
            if data.shape[1]-len(cat_cols)>0:# 有连续数字特征
                self.num_cols=list(set(range(data.shape[1])).difference(set(cat_cols)))
                data_num=d_c[:,self.num_cols]# 数字data
                self.means.append(np.mean(data_num,axis=0))
                self.vars.append(np.var(data_num,axis=0))

    def predict(self,data):
        result=[]
        for line in data:# 每一行数据
            result_tmp=[]
            for c_index in range(len(self.target_unique)):# 对每种class
                p_final=1
                if len(self.cat_cols)>0:# 有类别数据
                    for cat_f in self.cat_cols:# 每个离散类别特征列
                        p_final=p_final*self.feature_pct[c_index][line[0,cat_f]]/self.c_pct[c_index]
                if data.shape[1]-len(self.cat_cols)>0:# 有连续数字特征
                    for num_f in range(len(self.num_cols)):
                        x=line[self.num_cols[num_f]]
                        miu=self.means[c_index][num_f]
                        cita=self.vars[c_index][num_f]
                        ex=math.exp(-(math.pow(x-miu,2)/(2*math.pow(cita,2))))
                        p_final=p_final*ex / (math.sqrt(2*math.pi) * self.vars[c_index][num_f])/self.c_pct[c_index]
                p_final=p_final*self.c_pct[c_index]
                result_tmp.append(p_final)
            result.append(self.target_unique[np.argmax(result_tmp)])
        return np.array(result)
from sklearn.metrics import f1_score,accuracy_score,recall_score,precision_score,confusion_matrix
def jscore(clf,tx,ty,vx,vy,average='micro'):
    t_pre_y=clf.predict(tx)
    v_pre_y=clf.predict(vx)
    r=pd.DataFrame({'precision':[precision_score(ty,t_pre_y,average=average),precision_score(vy,v_pre_y,average=average)],
              'recall':[recall_score(ty,t_pre_y,average=average),recall_score(vy,v_pre_y,average=average)],
              'accuracy':[accuracy_score(ty,t_pre_y),accuracy_score(vy,v_pre_y)],
              'f1':[f1_score(ty,t_pre_y,average=average),f1_score(vy,v_pre_y,average=average)]},
            index=['train','test'])
    print('Train: \r\n'+str(confusion_matrix(ty,t_pre_y)))
    print('Val: \r\n'+str(confusion_matrix(vy,v_pre_y)))
    return r   

if __name__ == "__main__":
    # '''
    # iris花卉数据，分类使用。样本数据集的特征默认是一个(150, 4)大小的矩阵，
    # 样本值是一个包含150个类标号的向量，包含三种分类标号。
    # '''
    
    from sklearn.datasets import load_iris

    iris = load_iris()
    data = iris.data
    target = iris.target

    from sklearn.model_selection import StratifiedShuffleSplit
    split=StratifiedShuffleSplit(n_splits=1,test_size=0.25,random_state=42)
    for train_index,val_index in split.split(data,target):
        s_train_x=data[train_index,:]
        s_train_y=target[train_index]
        s_val_x=data[val_index,:]
        s_val_y=target[val_index]
    
    j_clf=NaiveBayesJ()
    j_clf.fit(s_train_x,s_train_y)
    j_clf_y=j_clf.predict(s_val_x)
    print('j_clf Score:')
    print(jscore(j_clf,s_train_x,s_train_y,s_val_x,s_val_y))

    from sklearn.naive_bayes import GaussianNB
    sk_clf=GaussianNB()
    sk_clf.fit(s_train_x,s_train_y)
    sk_clf_y=sk_clf.predict(s_val_x)
    print('sk_clf Score:')
    print(jscore(sk_clf,s_train_x,s_train_y,s_val_x,s_val_y))
    print('----------')
    print('real y:')
    print(s_val_y)
    print('j y:')
    print(j_clf_y)
    print('sklearn y:')
    print(sk_clf_y)