#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: amrit
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score,train_test_split,cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
label_encoder = preprocessing.LabelEncoder() 
def encode(data):
    for col in data.columns:
        if data.dtypes[col] == "object":
            le = preprocessing.LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
    return data

data = pd.read_csv('/home/amrit/Downloads/movie_metadata.csv')
print("*********************************************************************")
print("Columns the Data have  : ")
print(data.columns)
print("*********************************************************************")
print("Graph to Show count of null or NaN values : ")
X=[]
Y=[]
for i in data.columns:
    X.append(i)
    Y.append(data[i].isna().sum())
x = range(len(X))
plt.bar(x, Y, color='green', align='center')
plt.title('Count of Null or Nan Values')
plt.xticks(x, X, rotation='vertical')
plt.show()
data['gross'].fillna((round(data['gross'].mean())), inplace=True)
data['budget'].fillna((round(data['budget'].mean())), inplace=True)
data['aspect_ratio'].fillna((round(data['aspect_ratio'].mean())), inplace=True)
data['content_rating'].fillna(data['content_rating'].value_counts().idxmax(),inplace=True)
data['num_critic_for_reviews'].fillna((round(data['num_critic_for_reviews'].mean())), inplace=True)
data['duration'].fillna(data['duration'].value_counts().idxmax(),inplace=True)
data['director_facebook_likes'].fillna(data['director_facebook_likes'].value_counts().idxmax(),inplace=True)
data['actor_3_facebook_likes'].fillna(data['actor_3_facebook_likes'].value_counts().idxmax(),inplace=True)
data['plot_keywords'].fillna(data['plot_keywords'].value_counts().idxmax(),inplace=True)
data['actor_1_name'].fillna(data['actor_1_name'].value_counts().idxmax(),inplace=True)
data['actor_3_name'].fillna(data['actor_3_name'].value_counts().idxmax(),inplace=True)
data['title_year'].fillna(data['title_year'].value_counts().idxmax(),inplace=True)
data['actor_2_facebook_likes'].fillna(data['actor_2_facebook_likes'].value_counts().idxmax(),inplace=True)
data['color'].fillna(data['color'].value_counts().idxmax(),inplace=True)
data['actor_2_name'].fillna(data['actor_2_name'].value_counts().idxmax(),inplace=True)
data['actor_1_facebook_likes'].fillna(data['actor_1_facebook_likes'].value_counts().idxmax(),inplace=True)
data['facenumber_in_poster'].fillna(data['facenumber_in_poster'].value_counts().idxmax(),inplace=True)
data['num_user_for_reviews'].fillna(data['num_user_for_reviews'].value_counts().idxmax(),inplace=True)
data['language'].fillna(data['language'].value_counts().idxmax(),inplace=True)
data['country'].fillna(data['country'].value_counts().idxmax(),inplace=True)
data['director_name'].fillna(data['director_name'].value_counts().idxmax(),inplace=True)
genres = data['genres']
imdb = data['imdb_score']
num_cr = data['num_critic_for_reviews']
dur = data['duration']
gen = defaultdict(list)
prof = []
for i,j in enumerate(genres):
    g = j.split("|")
    prof.append(data['gross'][i]-data['budget'][i])
    for k in g:
        gen[k].append(imdb[i])
print("*********************************************************************")
print("Graph to show Average IMDB score for each genre : ")
X=[]
Y=[]
for i in gen.keys():
    X.append(i)
    Y.append(sum(gen[i])/len(gen[i]))

x = range(len(X))
plt.bar(x, Y, color='green', align='center')
plt.title('average imdb score for each genre')
plt.xticks(x, X, rotation='vertical')
plt.show()
print("*********************************************************************")
print("Heat Map for showing Correlation between attributes : ")
data['profit'] = prof
corr = data.corr()
f, ax = plt.subplots(figsize =(9, 8)) 
sns.heatmap(corr, ax = ax, cmap ="YlGnBu", linewidths = 0.1) 
plt.show()
print("*********************************************************************")
print("Correlation of each attribute with target attribute IMDB score  : ")
print(corr['imdb_score'])
X=[]
Y=[]
for i in data['color']:
    if(i not in X):
        X.append(i)
        Y.append(1)
    else:
        Y[X.index(i)] = Y[X.index(i)]+1
print("*********************************************************************")
print("Graph showing Color values are not that useful : ")
x = range(len(X))
plt.bar(x, Y, color='green', align='center')
plt.title('color values')
plt.xticks(x, X, rotation='vertical')
plt.show()
data = data.drop(columns=['color'], axis = 1)

X = []
Y = []
gen = []
for i,j in enumerate(prof):
    gen.append([data['movie_title'][i],j])
gen = sorted(gen, key = lambda x: x[1],reverse= True)
print("*********************************************************************")
print("Graph shwoing top 30 movie according to imdb score : ")
for i in range(30):
    X.append(gen[i][0])
    Y.append(gen[i][1])
x = range(len(X))
plt.bar(x, Y, color='green', align='center')
plt.title('Top 30 movie according to imdb score')
plt.xticks(x, X, rotation='vertical')
plt.show()
gen = []
for i,j in enumerate(data['num_critic_for_reviews']):
    gen.append([data['movie_title'][i],j])
print("*********************************************************************")
print("Graph showing top 30 movies according to number of critics for review : ")
gen = sorted(gen, key = lambda x: x[1],reverse= True)
X=[]
Y=[]
for i in range(30):
    X.append(gen[i][0])
    Y.append(gen[i][1])
x = range(len(X))
plt.bar(x, Y, color='green', align='center')
plt.title('Top 30 movie according to number of critic for review')
plt.xticks(x, X, rotation='vertical')
plt.show()
data = data.drop(columns=['facenumber_in_poster','title_year'],axis=1)
data = data.drop_duplicates()
label = data['imdb_score']
label_1 = []
for i in label:
    if(i >0 and i<=2):
        label_1.append(1)
    elif(i>2 and i<= 4):
        label_1.append(2)
    elif(i>4 and i<=6):
        label_1.append(3)
    elif(i>6 and i<=8):
        label_1.append(4)
    else:
        label_1.append(5)
X=[]
Y=[]  
data = data.drop(columns=['imdb_score'],axis=1)
print("*********************************************************************")
print("Columns after dropping all non-required attributes : ")
print(data.columns)
data = encode(data)
X_train,X_test,Y_train,Y_test = train_test_split(data,label_1,test_size=0.20,random_state=0)
print("*********************************************************************")
print("Graph Showing accuracy v/s k in knn : ")
for i in range(5,20):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(X_train,Y_train)
    check = knn.predict(X_test)
    X.append(i)
    Y.append(accuracy_score(check,Y_test))
plt.plot(X,Y)
plt.xlabel('Value of k in k-nearest neighbors') 
plt.ylabel('Accuracy') 
plt.title('Accuracy of kNN vs value of k in k-NN') 
plt.show() 
print("*********************************************************************")
print("Accuracy of KNN : ")
print("Maximum accuracy of k-NN is : ",max(Y))
X=[]
Y=[]  
ac=[]
for i in range(2,10):
    clf = DecisionTreeClassifier(random_state=0,max_depth=i)
    clf.fit(X_train,Y_train)
    scores = cross_validate(clf, X_train, Y_train, cv=5,scoring=('accuracy'),return_train_score=True,error_score ='raise',return_estimator=True)
    s = cross_val_score(clf,X_train,Y_train,cv=5)
    Y.append(s.mean())
    X.append(i)
    check = clf.predict(X_test)
    ac.append(accuracy_score(Y_test,check))
print("*********************************************************************")
print("Graph to show Train accuracy of Decision Tree v/s Depth of tree : ")
plt.plot(X,Y)
plt.xlabel('Depth of the tree (k)') 
plt.ylabel('Train accuracy') 
plt.title('Train accuracy vs Depth of the tree in k-fold validation') 
plt.show() 
plt.plot(X, ac) 
print("*********************************************************************")
print("Graph to show Test accuracy of Decision Tree v/s Depth of tree : ")
plt.xlabel("Depth of the tree (k)")
plt.ylabel("Test accuracy")
plt.title('Test accuracy vs Depth of the tree in k-fold validation') 
plt.show()
print("*********************************************************************")
print("Accuracy of Decision Tree implementation with 80:20 train-test split : ")
print(sum(ac)/len(ac))
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train,Y_train)
y_label = svm_classifier(X_test)
print("*********************************************************************")
print("Accuracy of SVM classifier implementation with 80:20 train-test split : ")
print(accuracy_score(Y_test,y_label))

