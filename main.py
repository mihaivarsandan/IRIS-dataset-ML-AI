from tkinter import X
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline,make_pipeline
from Majority_Voting_Classifier import  MajorityVoteClassfier
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV

"""Loading the dataset and splitting it into train,test"""
le = LabelEncoder()
iris = datasets.load_iris()
x = iris.data
y = iris.target
y = le.fit_transform(y)

#print('Class labels:',np.unique(y))
X_train, X_test, y_train, y_test=train_test_split(x,y,test_size=0.3
                                ,random_state=1,stratify=y)

"""Classifiers"""
# Support Vector Machine
clf1 = SVC(kernel='linear',C=1.0, random_state=1,decision_function_shape='ovo',probability=True)
# Decision Tree
clf2 = DecisionTreeClassifier(max_depth=1,criterion='entropy',random_state=0)
# K-nearest neighbours
clf3 = KNeighborsClassifier(n_neighbors=1,p=2,metric='minkowski')
# Logistic Regression
clf4 = LogisticRegression(penalty='l2',C=0.001,random_state=1,multi_class='ovr')

pipe1 = make_pipeline(StandardScaler(),clf1)
pipe3 = make_pipeline(StandardScaler(),clf3)
pipe4 = make_pipeline(StandardScaler(),clf4)
mv_clf = MajorityVoteClassfier(classifiers=[pipe1,clf2,pipe3,pipe4])
clf_labels = ['SVM','Decision Tree','KNN','Majority voting','Logistic Regression']


"""Grid Search"""
params = {'decisiontreeclassifier__max_depth':[1,2],
        'pipeline-3__logisticregression__C':[0.001,0.1,100.0],
        'pipeline-1__svc__C':[0.001,0.1,100.0]}



grid = GridSearchCV(estimator=mv_clf,param_grid=params,cv=10,scoring='roc_auc_ovo')
grid.fit(X_train,y_train)


for i in range(len(grid.cv_results_['params'])): 
    print('%0.3f +/- %0.2f %r' %(grid.cv_results_['mean_test_score'][i],grid.cv_results_['std_test_score'][i]/2,grid.cv_results_['params'][i]))

clf = grid.best_estimator_
print(grid.best_score_)
print(grid.best_params_)
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
print(y_test-y_pred)
