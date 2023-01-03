#!/usr/bin/env python
# coding: utf-8

import numpy as np
from math import sqrt
import pandas as pd

Array=pd.DataFrame(data=None,columns=['Name','Score','AD','AP','Mana','Speed','Strike','Arma','Resistance','HP','Jobs'])


# In[27]:


Array.loc[0]=['平民枪手 莎米拉[破甲、无尽、巨杀]，扎克[肉]',100,20,0,0,20,20,30,10,20,0]
Array.loc[1]=['平民枪手 莎米拉[破甲、杀人刀、分裂弓]，扎克[肉]',90,20,0,0,20,10,30,20,20,0]
Array.loc[2]=['怪兽枪手 厄斐琉斯[破甲、巨杀、羊刀]，艾克[肉]',100,0,10,10,30,10,30,10,20,0]
Array.loc[3]=['怪兽枪手 厄斐琉斯[破甲、分裂弓、正义]，艾克[肉]',85,0,0,10,20,20,30,20,20,0]
Array.loc[4]=['灵能岩雀 塔利亚[青龙刀、法爆、科技枪],安妮[肉]',100,20,20,10,0,10,40,10,10,0]
Array.loc[5]=['灵能岩雀 塔利亚[青龙刀、法爆、破防者]，安妮[肉]',95,10,10,10,0,20,40,10,20,0]
Array.loc[6]=['高斗武器 贾克斯[火炮、羊刀、吸血]，瑞文[肉]',80,10,20,10,30,0,10,20,20,0]
Array.loc[7]=['高斗武器 贾克斯[火炮、羊刀、巨杀]，瑞文[肉]',80,10,20,10,40,0,10,10,20,0]
Array.loc[8]=['怪兽卡莎 卡莎[电刀、科技枪、巨杀]，龙龟[肉]',80,20,10,20,20,0,20,10,20,0]
Array.loc[9]=['怪兽卡莎 卡莎[电刀、羊刀、破防者]，龙龟[肉]',85,0,10,20,20,10,20,10,30,0]
Array.loc[10]=['怪兽塞纳 塞纳[破甲、无尽、巨杀]，龙龟[肉]',80,20,0,10,20,20,20,10,20,0]
Array.loc[11]=['怪兽塞纳 塞纳[破甲、巨杀、羊刀]，龙龟[肉]',85,20,10,10,20,10,20,10,20,0]
Array.loc[12]=['怪兽佐伊 佐伊[蓝、鬼书、帽子]，龙龟[肉]',90,0,40,20,0,0,20,10,30,0]
Array.loc[13]=['怪兽佐伊 佐伊[蓝、鬼书、巨杀]，龙龟[肉]',80,10,20,20,10,0,20,10,30,0]
Array.loc[14]=['传统佐伊 佐伊[蓝、鬼书、法爆]，安妮[肉]',85,0,20,20,10,10,30,0,30,0]
Array.loc[15]=['传统佐伊 佐伊[蓝、鬼书、科技枪]，安妮[肉]',86,10,30,20,10,0,10,10,30,0]


# In[28]:


class KNNClassifier:
    def __init__(self, k):
        #初始化kNN分类器
        assert k >= 1, "k must be valid"
        self.k = k
        self._X_train = None
        self._y_train = None
    
    def fit(self, X_train, y_train):#商品类型，分类的指向
        assert X_train.shape[0]==y_train.shape[0],"the size of X_train must equal to the size of y_train"
        assert self.k <= X_train.shape[0],"the size of X_train must be at least k."
        self._X_train = X_train
        self._y_train = y_train
        return self
    
    def predict(self, X_predict):#给定待预测数据集X_predict，返回标示X_predict的结果向量
        assert self._X_train is not None and self._y_train is not None,"mush fit before predict"
        assert self._X_train.shape[1] == X_predict.shape[1],"the feature number of x must be equal to X_train"
        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)
    
    def _predict(self, x):#给定单个待预测数据x，返回x的预测结果值
        distances = [sqrt(np.sum((x_train - x) ** 2)) for x_train in self._X_train]
        nearset = np.argsort(distances)
        top_y = [self._y_train[i] for i in nearset[:self.k]]
        top_distances=[distances[i] for i in nearset[:self.k]]
        top_score=[(120-i)/120*100 for i in top_distances]
        return top_y,top_score


# In[29]:


x_train=np.array(Array.iloc[:,2:])
y_train=np.array(Array.iloc[:,0])


# In[30]:


while True:
    try:
        input_str=input('按[大剑，大棒，水滴，攻速，暴击，护甲，魔抗，腰带，铲子]顺序输入数量（共9位，空格分割）：')
        tokens = input_str.split()
        tokens=[int(x)*10 for x in tokens]
        if len(tokens)==9:
            x_predict=np.array(tokens)
            break
    except:
        print('输入有误，请重新输入：')


# In[31]:


kNN_classifier = KNNClassifier(5)
kNN_classifier.fit(x_train, y_train)
x_predict=x_predict.reshape(1,-1)
y_predict=kNN_classifier.predict(x_predict)


# In[32]:


y=pd.DataFrame(data=[y_predict[0,0],y_predict[0,1]]).T
y.rename(columns={0:'Name',1:'装备适应度'},inplace=True)


# In[33]:


y=pd.merge(y,Array[['Name','Score']],on=['Name'])
y.rename(columns={'Name':'阵容名','Score':'阵容分'},inplace=True)


# In[34]:


y['总分']=y['装备适应度'].astype('float64')+y['阵容分']


# In[35]:


output=y.sort_values(by='总分',ascending=False).head(3)
output.reset_index(drop=True,inplace=True)
print(output)

input('Press <Enter>')

# In[24]:





# In[36]:



# In[ ]:




