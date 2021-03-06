#!/usr/bin/python3
# coding=utf-8

import numpy as np
import pandas as pd
import math
from sklearn.base import BaseEstimator
class CartClassifierJ(BaseEstimator):
    # cart分类树
    def __init__(self,class_weight=None,cat_cols=[],min_samples_split=2,min_samples_leaf=1,min_weight_fraction_leaf=0,min_impurity_split=1e-7):
        self.trees={}
        self.tree_id=0
        self.trees_togrow=[]
        # self.trees_growed=[]
        self.n_samples=None
        self.min_samples_split=min_samples_split
        self.min_samples_leaf=min_samples_leaf
        self.cat_cols=cat_cols
        self.class_weight={}
        if class_weight:
            self.class_weight=class_weight# dic
        self.min_weight_fraction_leaf=min_weight_fraction_leaf
        self.min_impurity_split=min_impurity_split
        self.sample_weight=None
        pass
    
    def ctree(self,list_data_target):
        # list_data_target---
        # 0:tree_id
        # 1:p_data
        # 2:p_target
        # 3:p_sample_weight
        # 4:y_pred
        # 5:prob
        p_data=list_data_target[1]
        p_target=list_data_target[2]
        p_sample_weight= list_data_target[3]
        if (p_data.shape[0]<self.min_samples_split if self.min_samples_split>1 else p_data.shape[0]<self.min_samples_split*self.n_samples) or \
            (p_data.shape[0]<self.min_samples_leaf*2 if self.min_samples_leaf>=1 else p_data.shape[0]<self.min_samples_leaf*self.n_samples) or \
            (len(np.unique(p_target))<2) or \
            np.sum(p_sample_weight)<=2*self.min_weight_fraction_leaf:
            # 节点样本数小于2（min_samples_split）or 节点样本数 < 样本总数*最小比例
            # 样本数小于2*叶节点最小样本数（min_samples_leaf）or 节点样本数 < 2*样本总数*最小比例
            # 类别只有1类
            # 节点权重<=2*min_weight_fraction_leaf

            # 每种类别
            p=[np.sum(p_sample_weight[p_target==i]) for i in np.unique(p_target)]
            # self.trees_growed.append[list_data_target.append(np.unique(p_target)[np.argmax(p)]).append(p)]
            # id:[gini_da,s_feat,s_num,left_id,right_id,y,p]
            self.trees[list_data_target[0]]=[-1,-1,-1,-1,-1]+[np.unique(p_target)[np.argmax(p)],p]
            self.trees_togrow.remove(list_data_target)
            pass
        else:
            cat_data=None
            num_data=None
            list_best_split=[]
            target_unique=np.unique(p_target)
            if len(self.cat_cols)>=1:# 有类别特征
                cat_data=p_data[:,self.cat_cols]
            if len(self.cat_cols)<p_data.shape[1]:# 有连续特征
                num_data=p_data[:,list(set(list(range(p_data.shape[1]))).difference(set(self.cat_cols)))]
            list_ginida_nf_sp=[]
            # for cf in range()
            for nf in range(num_data.shape[1]):# 每个连续特征
                argsort_nf=np.argsort(num_data[:,nf])# 这个特征的argsort
                for rank_t in range(num_data.shape[0]-1):# 测试这个特征的每个特征值-分裂点
                    d1_data=num_data[:,nf][argsort_nf<=rank_t]
                    d1_target=p_target[argsort_nf<=rank_t]
                    d1_sample_weight=p_sample_weight[argsort_nf<=rank_t]
                    sum_ginik=0
                    weight_list=[]
                    for i in np.unique(d1_target):
                        weight_list.append(np.sum(d1_sample_weight[d1_target==i]))
                    weight_sum1=sum(weight_list)
                    # for k in target_unique:# 对1个目标分类k来说
                    #     is_k=np.sum(d1_target==k)/(rank_t+1)
                    #     sum_ginik+=is_k**2
                    # gini_d1=1-sum_ginik
                    for k in np.unique(d1_target):# 对1个目标分类k来说
                        is_k=np.sum(d1_sample_weight[d1_target==k])/weight_sum1
                        sum_ginik+=is_k**2
                    gini_d1=1-sum_ginik
                    
                    d2_data=num_data[:,nf][argsort_nf>rank_t]
                    d2_target=p_target[argsort_nf>rank_t]
                    d2_sample_weight=p_sample_weight[argsort_nf>rank_t]
                    sum_ginik=0
                    weight_list=[]
                    for i in np.unique(d2_target):
                        weight_list.append(np.sum(d2_sample_weight[d2_target==i]))
                    weight_sum2=sum(weight_list)
                    for k in np.unique(d2_target):# 对1个目标分类k来说
                        is_k=np.sum(d2_sample_weight[d2_target==k])/weight_sum2
                        sum_ginik+=is_k**2
                    gini_d2=1-sum_ginik

                    # gini_da=(rank_t+1)/num_data.shape[0]*gini_d1+(d2_target.shape[0])/num_data.shape[0]*gini_d2
                    gini_da=weight_sum1/np.sum(p_sample_weight)*gini_d1+weight_sum2/np.sum(p_sample_weight)*gini_d2
                    list_ginida_nf_sp.append([gini_da,nf,num_data[:,nf][argsort_nf==rank_t][0]])
            ndarray_ginida_nf_sp=np.array(list_ginida_nf_sp)
            best_split=ndarray_ginida_nf_sp[np.argmin(ndarray_ginida_nf_sp[:,0]),:]
            list_best_split.append(best_split.tolist())
            print('TreeID='+str(list_data_target[0])+'，最佳分裂：gini_da='+str(best_split[0])+'，数字特征='+str(int(best_split[1]))+'，分裂点≤'+str(best_split[2])+'，左树ID='+str(self.tree_id+1)+'，右树ID='+str(self.tree_id+2))# 正常分裂点应该取两个值的中间值，这里直接取了样本特征值
            self.trees[list_data_target[0]]=best_split.tolist()+[self.tree_id+1,self.tree_id+2]
            lines_left=p_data[:,int(best_split[1])]<=best_split[2]
            left_data=p_data[lines_left,:]
            right_data=p_data[~(lines_left),:]
            left_sample_weight=p_sample_weight[lines_left]
            right_sample_weight=p_sample_weight[~(lines_left)]
            if left_data.shape[0]==0 or right_data.shape[0]==0:
                p=[np.sum(p_sample_weight[p_target==i]) for i in np.unique(p_target)]
                self.trees[list_data_target[0]]=[-1,-1,-1,-1,-1]+[np.unique(p_target)[np.argmax(p)],p]
            else:
                self.trees_togrow.append([self.tree_id+1,left_data,p_target[lines_left],left_sample_weight])
                self.trees_togrow.append([self.tree_id+2,right_data,p_target[~(lines_left)],right_sample_weight])
                self.tree_id+=2
            self.trees_togrow.remove(list_data_target)
            # self.trees={}
            # id:[gini_da,s_feat,s_num,left_id,right_id,y,p]
            pass
    def fit(self,data,target,sample_weight=None):
        # sample_weight:array
        self.n_samples=target.shape[0]
        if not sample_weight:# array
            # sample_weight=np.array(([1/self.n_samples]*self.n_samples))
            sample_weight=np.array(([1]*self.n_samples))
        if not self.class_weight:
            for i in np.unique(target):
                self.class_weight[i]=1-np.sum(target==i)/self.n_samples 
        for i in range(self.n_samples):
            sample_weight[i]=sample_weight[i]*self.class_weight[target[i]]
        self.sample_weight=sample_weight

        self.trees_togrow.append([self.tree_id,data,target,self.sample_weight])
        while True:
            if len(self.trees_togrow)>0:
                self.ctree(self.trees_togrow[0])
            else:
                break
        # for t in self.trees_togrow:
        print('fit done')
        pass

    def predict(self,data):
        list_result=[]
        list_prob=[]
        for i in data:
            t_id=0
            while True:
                t=self.trees[t_id]
                if len(t)==7:#叶子结点
                    list_result.append(t[5])
                    list_prob.append(t[6])
                    break
                else:
                    if i[int(t[1])]<=t[2]:
                        t_id=t[3]
                    else:
                        t_id=t[4]
        # return np.array(list_result),list_prob
        return np.array(list_result)
        pass


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
        
    # j_clf=CartClassifierJ(class_weight=None,cat_cols=[],min_samples_split=2,min_samples_leaf=1,min_weight_fraction_leaf=0,min_impurity_split=1e-7)
    j_clf=CartClassifierJ(class_weight={0:1,1:1,2:1},cat_cols=[],min_samples_split=2,min_samples_leaf=1,min_weight_fraction_leaf=0,min_impurity_split=1e-7)
    j_clf.fit(s_train_x,s_train_y)
    j_clf_y=j_clf.predict(s_val_x)
    print('j_clf Score:')
    print(jscore(j_clf,s_train_x,s_train_y,s_val_x,s_val_y))

    from sklearn.tree import DecisionTreeClassifier
    sk_clf=DecisionTreeClassifier()
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
