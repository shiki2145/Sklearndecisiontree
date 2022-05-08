'''
==================
test案例
==================
'''
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np
pd.set_option('display.max_columns', None)
#
traindata = pd.read_csv(r"C:\Users\86177\Desktop\test\dess\data-train.csv")#这里可以添加自己的数据集
testdata=pd.read_csv(r"C:\Users\86177\Desktop\test\dess\data-test.csv")#这里可以添加自己的数据集
#也可以选择sklearn中自带的数据集
# print(data)

#以下都是在训练集上运行的，调好参数再去测试集，记住别直接在测试集运行，搞个备份
x = traindata.iloc[:, traindata.columns != "target"]#确定label和非label
y = traindata.iloc[:, traindata.columns == "target"]
# print(x)
# print(y)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(x, y, test_size=0.3)
# print(Xtest)

# for i in [Xtrain, Xtest, Ytrain, Ytest]:
#     i.index = range(i.shape[0])
'''-------以下三块代码不要同时运行，会很慢，在第三块找到合适的模型参数后可以回来运行下面这块---------'''
clf = DecisionTreeClassifier(criterion="gini"
                             ,max_depth=3
                             ,splitter="best"
                             ,min_samples_leaf=6
                             ,min_samples_split=6
                             ,random_state=25)
clf = clf.fit(Xtrain, Ytrain)
score = clf.score(Xtest,Ytest)
clf2 = DecisionTreeClassifier(random_state=30)
score2 = cross_val_score(clf2, x, y, cv=10)
print(score)
print(score2)
#找好参数后来测试集运行
# test_x = testdata.iloc[:, testdata.columns != "target"] #确定数据
# test_y = clf.predict(test_x) #把预测结果赋值给y
# test_x.insert(13,'target',test_y) #拼一起看一看
# print(test_x)
# 接下来就是把y写入csv文件里面了，蛮简单的（摸了

'''-------下面这段可以画图看效果（一般般），时间太久远我也有点忘了，但是可以直接运行---------'''
# tr = []
# te = []
# for i in range(10):#通过i来改变参数值，下面是检测max_depth的例子
#     clf = DecisionTreeClassifier(random_state=25
#                                  , max_depth=i+1
#                                  , criterion="gini"
#                                  , splitter="best"
#                                  , min_samples_leaf=3
#                                  , min_samples_split=4
#                                  )
#     clf = clf.fit(Xtrain, Ytrain)
#     score_tr = clf.score(Xtrain, Ytrain)
#     score_te = cross_val_score(clf, x, y, cv=10).mean()
#     tr.append(score_tr)
#     te.append(score_te)
# print(max(te))
# plt.plot(range(1,11), tr, color="red", label="train")
# plt.plot(range(1,11), te, color="blue", label="test")
# plt.xticks(range(1,11))
# plt.legend()
# plt.show()
'''----------下面这块代码是为了找到更好的参数的，会输出分数（虽然每次都不太一样，参考着用）---------'''
# gini_threholds = np.linspace(0,0.5,50)
# parameters = {"criterion":("gini","entropy")
#              , "splitter":("best","random")
#              , "max_depth":[*range(1,6)]
#              , "min_samples_leaf":[*range(1,60,3)]
#              # ,"min_samples_split":[*range(4,8)]
#               }#这些参数不一定要全用，有时候少一个反而更好，深度什么的也不是越深越好，容易过拟合
# clf = DecisionTreeClassifier(random_state=25)
# GS = GridSearchCV(clf, parameters, cv=10) #这是对模型进行的参数检测
# GS = GS.fit(Xtrain, Ytrain)
# a = GS.best_params_#输出最佳参数值
# b = GS.best_score_#输出分数
# print(a)
# print(b)
