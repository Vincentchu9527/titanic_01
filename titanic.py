# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import cross_val_score

# 数据加载
train_data = pd.read_csv('./Data/train.csv')
test_data = pd.read_csv('./Data/test.csv')

# 数据探索
print(train_data.info())
print(train_data.describe(include=['O']))     # 使用 describe(include=[‘O’]) 查看字符串类型（非数字）的整体情况
print(train_data.head())
print(test_data.info())
print(test_data.describe(include=['O']))
print(test_data.head())

# 使用平均年龄来填充年龄中的 nan 值
train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
test_data['Age'].fillna(test_data['Age'].mean(), inplace=True)
# 使用票价的均值填充票价中的 nan 值
train_data['Fare'].fillna(train_data['Fare'].mean(), inplace=True)
test_data['Fare'].fillna(test_data['Fare'].mean(), inplace=True)
# 使用登录最多的港口来填充登录港口的 nan 值
train_data['Embarked'].fillna('S', inplace=True)
test_data['Embarked'].fillna('S', inplace=True)

# 特征选择
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
train_features = train_data[features]
train_labels = train_data['Survived']
test_features = test_data[features]

# 1. 使用字典特征提取器处理符号化对象
dvec = DictVectorizer(sparse=False)
train_features = dvec.fit_transform(train_features.to_dict(orient='record'))
test_features = dvec.fit_transform(test_features.to_dict(orient='record'))
print(dvec.feature_names_)

# # 2. 使用pd.get_dummies() 进行独热编码
# train_features = pd.get_dummies(train_features)
# test_features = pd.get_dummies(test_features)
# print(train_features.head())
# print(test_features.head())

# # 3. 使用pd.factorize() 转换
# sex_res1 = pd.factorize(train_features['Sex'])
# sex_res2 = pd.factorize(test_features['Sex'])
# embarked_res1 = pd.factorize(train_features['Embarked'])
# embarked_res2 = pd.factorize(test_features['Embarked'])
# print(sex_res1)
# print(sex_res2)
# print(embarked_res1)  # Index(['S', 'C', 'Q']
# print(embarked_res2)  # Index(['Q', 'S', 'C']
# 如果index 对应, 可执行以下操作
# train_features['Sex'] = sex_res1[0]
# train_features['Embarked'] = embarked_res1[0]
# test_features['Sex'] = sex_res2[0]
# test_features['Embarked'] = embarked_res2[0]
# print(train_features.head(10))

# 构造 ID3 决策树
clf = DecisionTreeClassifier(criterion='entropy')

# 决策树训练和预测
clf.fit(train_features, train_labels)
pre = clf.predict(test_features)

# 得到决策树训练集准确率
acc_decision_tree = round(clf.score(train_features, train_labels), 6)
print(u'score 准确率为 %.4lf' % acc_decision_tree)

# 使用 K 折交叉验证 统计决策树准确率
print(u'cross_val_score 准确率为 %.4lf' % np.mean(cross_val_score(clf, train_features, train_labels, cv=10)))

# 查看结果并储存
test_data['Survived'] = pre
print(test_data.head(10))
test_data.to_excel('./result/titanic_res.xls')