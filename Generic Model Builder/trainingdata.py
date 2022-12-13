#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
import numpy as np
#from scipy.stats import norm
#from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, auc, precision_recall_curve, mean_squared_error, r2_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
import pickle
from pandas import Series
from xgboost import XGBClassifier
from datetime import datetime
from sklearn.metrics import precision_score
import statsmodels.api as sm

    # Function to visualise classes in Y
def class_count(df, output_col):
    plt.figure()
    df[output_col].value_counts().plot(kind = 'bar')
    plt.ylabel("Count")
    plt.title('ORIGINAL CLASS COUNT')
    
    # Function for visualising important features of the dataset
def imp_feat(df, model):
    features_list = df.columns.values
    print(features_list)
    feature_importance = model.feature_importances_
    print('printing importance features')
    print(feature_importance)
    
    sorted_idx = np.argsort(feature_importance)[:20]
    plt.figure(figsize = (5,7))
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), features_list[sorted_idx])
    plt.xlabel('Importance')
    plt.title('Feature importances')
    plt.draw()
    plt.show()
    #exit(0)

 # second function  
def imp_feat2(df, model):
    features_list = df.columns.values
    print(features_list)
    feature_importance = model.coef_
    print('printing importance features')
    print(feature_importance)
    
    sorted_idx = np.argsort(feature_importance)[:20]
    plt.figure(figsize = (5,7))
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), features_list[sorted_idx])
    plt.xlabel('Importance')
    plt.title('Feature importances')
    plt.draw()
    plt.show()
    
# Function to show the cross validation score
def scoring(model, X, Y):
    scores = cross_val_score(model, X, Y, cv=5)
    print(scores.mean())

# Function for Support Vector Regression    
    
def SVR(X_train, Y_train, X_test, Y_test, model_xml, model_s, accept_acc, X, Y,df_train):
    from sklearn.svm import SVR
    tree1 = ET.parse(model_xml)
    root1 = tree1.getroot()
    reg = root1[0].text
    ker = root1[1].text
    svr = SVR()
    parameters = svr.get_params()
    if reg != ' ':
        reg = float(reg)
    else:
        reg = parameters['C']
    if ker != ' ':
        ker = str(ker)
    else:
        ker = parameters['kernel']
    svr = SVR(C = reg, kernel = ker)
    svr.fit(X_train, Y_train)
    pickle.dump(svr, open(model_s, 'wb'))
    imp_feat(df_train,svr)
    Y_pred = svr.predict(X_test)
    accuracy = r2_score(Y_test, Y_pred)
    print('SV Regressor accuracy: %0.3f'% accuracy)
    if accuracy > accept_acc:
        svr.fit(X,Y)
        pickle.dump(svr, open(model_s, 'wb'))
        message = "Success"
    else:
        print('Advising to use another model as the accuracy for this one is', accuracy)  
        message = "Advising to use another model as the accuracy for this one is low"
        model_s = ""
    conf_matrix = []
    summ = ""
    summ = str(summ)
    model_s1 = model_s
    return accuracy, conf_matrix, summ, message, model_s1  
        

# Function for Random Forest Classifier
def RFC(X_train, Y_train, X_test, Y_test, model_xml, model_s, accept_acc, X, Y,df_train):
    from sklearn.ensemble import RandomForestClassifier
    print('Executing RF Classifier')
    tree1 = ET.parse(model_xml)
    root1 = tree1.getroot()
    nest = root1[0].text
    state = root1[1].text
    njobs = root1[2].text
    RF = RandomForestClassifier()
    parameters = RF.get_params()
    if nest != ' ':
        nest = int(nest)
    else:
        nest = parameters['n_estimators']
    if state != ' ':
        state = int(state)
    else:
        state = parameters['random_state']
    if njobs != ' ':
        njobs = int(njobs)
    else:
        njobs = parameters['n_jobs']
    RF = RandomForestClassifier(n_estimators = nest, random_state = state, n_jobs = njobs)
    RF.fit(X_train, Y_train)
    pickle.dump(RF, open(model_s, 'wb'))
    imp_feat(df_train,RF)
    Y_pred = RF.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    print('Random Forest Classification accuracy: %0.3f'% accuracy)
    print("Classifier report \n", classification_report(Y_test, Y_pred))
    if accuracy > accept_acc:
        RF.fit(X,Y)
        pickle.dump(RF, open(model_s, 'wb'))
        message = "Success"
    else:
        
        print('Advising to use another model as the accuracy for this one is', accuracy)
        message = "Advising to use another model as the accuracy for this one is low"
        model_s = ""
    conf_matrix = confusion_matrix(Y_test, Y_pred)
    classi_report = classification_report(Y_test, Y_pred)
    print(conf_matrix)
    print(classi_report)    
    model_s1 = model_s
    return accuracy, conf_matrix, classi_report, message, model_s1
    
# Function for Logictic Regression
def LOR(X_train, Y_train, X_test, Y_test, model_xml, model_s, accept_acc, X, Y,df_train):
    from sklearn.linear_model import LogisticRegression
    print('Executing Logistic Regression')
    tree1 = ET.parse(model_xml)
    root1 = tree1.getroot()
    inv_reg = root1[0].text
    state = root1[1].text
    print('Read XML file')
    LoR = LogisticRegression()
    parameters = LoR.get_params()
    if inv_reg != ' ':
        inv_reg = float(inv_reg)
    else:
        inv_reg = parameters['C']
    if state != ' ':
        state = int(state)
    else:
        state = parameters['random_state']
    LoR = LogisticRegression(C = inv_reg, random_state = state)
    LoR.fit(X_train, Y_train)
    print('Model has been fit to the data')
    pickle.dump(LoR, open(model_s, 'wb'))
    imp_feat2(df_train, LoR)
    Y_pred = LoR.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    print('Logistic Regression Classification accuracy: %0.3f'% accuracy)
    print("Classifier report \n", classification_report(Y_test, Y_pred))
    if accuracy > accept_acc:
        LoR.fit(X,Y)
        pickle.dump(LoR, open(model_s, 'wb'))
        message = "Success"
        
    else:
        print('Advising to use another model as the accuracy for this one is', accuracy)
        message = "Advising to use another model as the accuracy for this one is low"
        model_s = ""
    conf_matrix = confusion_matrix(Y_test, Y_pred)
    classi_report = classification_report(Y_test, Y_pred)
    print(conf_matrix)
    print(classi_report)   
    model_s1 = model_s
    return accuracy, conf_matrix, classi_report, message, model_s1

# Function for Linear Regression
def LR(X_train, Y_train, X_test, Y_test, model_xml, model_s, accept_acc, X, Y,df_train):
    from sklearn.linear_model import LinearRegression 
    # Returns a datetime object containing the local date and time
    dateTimeObj = datetime.now()
    tree1 = ET.parse(model_xml)
    root1 = tree1.getroot()
    njobs = root1[0].text
    LR = LinearRegression()
    parameters = LR.get_params()
    if njobs != ' ':
        njobs = int(lr)
    else:
        njobs = parameters['n_jobs']
    print(dateTimeObj)    
    LiR = LinearRegression(n_jobs = njobs)
    LiR.fit(X_train, Y_train)
    pickle.dump(LiR, open(model_s, 'wb'))
    Y_pred = LiR.predict(X_test)
    imp_feat2(df_train, LiR)
    
    accuracy = r2_score(Y_test, Y_pred)
    print('Linear Regression accuracy: %0.3f'% accuracy)
    if accuracy > accept_acc:
        LiR.fit(X,Y)
        pickle.dump(LiR, open(model_s, 'wb'))
        message = "Success"
        
    else:
        message = "Advising to use another model as the score for this one is low"
        print('Advising to use another model as the score for this one is', accuracy)
        model_s = ""
    conf_matrix = []
    # Add a constant to get an intercept
    X_train_sm = sm.add_constant(X_train)

    # Fit the resgression line using 'OLS'
    lr = sm.OLS(Y_train, X_train_sm).fit()   
    summ = lr.summary()
    #print(summ)
    summ = str(summ)
    print(summ)
    model_s1 = model_s
    return accuracy, conf_matrix, summ, message, model_s1  

# Function for Ada Boost Classifier
def ABC(X_train, Y_train, X_test, Y_test, model_xml, model_s, accept_acc, X, Y,df_train):
    from sklearn.ensemble import AdaBoostClassifier
    tree1 = ET.parse(model_xml)
    root1 = tree1.getroot()
    best = root1[0].text
    nest = root1[1].text
    lr = root1[2].text
    state = root1[3].text
    AB = AdaBoostClassifier()
    parameters = AB.get_params()
    if best != ' ':
        best = object(best)
    else:
        best = parameters['base_estimator']
    if nest != ' ':
        nest = int(nest)
    else:
        nest = parameters['n_estimators']
    if lr != ' ':
        lr = float(lr)
    else:
        lr = parameters['learning_rate']
    if state != ' ':
        state = int(state)
    else:
        state = parameters['random_state']
    AB = AdaBoostClassifier(base_estimator = best, n_estimators = nest, learning_rate = lr, random_state = state)
    AB.fit(X_train, Y_train)
    pickle.dump(AB, open(model_s, 'wb'))
    imp_feat(df_train,AB)
    Y_pred = AB.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    print('Ada Boost Classification accuracy: %0.3f'% accuracy)
    print("Classifier report \n", classification_report(Y_test, Y_pred))
    if accuracy > accept_acc:
        AB.fit(X,Y)
        pickle.dump(AB, open(model_s, 'wb'))
        message = "Success"
    else:
        print('Advising to use another model as the accuracy for this one is', accuracy)
        message = "Advising to use another model as the accuracy for this one is low"
        model_s = ""
    conf_matrix = confusion_matrix(Y_test, Y_pred)
    classi_report = classification_report(Y_test, Y_pred)
    print(conf_matrix)
    print(classi_report)    
    model_s1 = model_s
    return accuracy, conf_matrix, classi_report, message, model_s1  

# Function for Gradient Boost Classifier
def GBC(X_train, Y_train, X_test, Y_test, model_xml, model_s, accept_acc, X, Y,df_train):
    from sklearn.ensemble import GradientBoostingClassifier
    tree1 = ET.parse(model_xml)
    root1 = tree1.getroot()
    lr = root1[0].text
    nest = root1[1].text
    state = root1[2].text
    GB = GradientBoostingClassifier()
    parameters = GB.get_params()
    if lr != ' ':
        lr = float(lr)
    else:
        lr = parameters['learning_rate']
    if nest != ' ':
        nest = int(nest)
    else:
        nest = parameters['n_estimators']
    if state != ' ':
        state = int(state)
    else:
        state = parameters['random_state']
    GB = GradientBoostingClassifier(learning_rate = lr, n_estimators = nest, random_state = state)
    GB.fit(X_train, Y_train)
    pickle.dump(GB, open(model_s, 'wb'))
    imp_feat(df_train,GB)
    Y_pred = GB.predict(X_test)  
    accuracy = accuracy_score(Y_test, Y_pred)
    print('Classifier accuracy: %0.3f'% accuracy)
    print("Classifier report \n", classification_report(Y_test, Y_pred))
    if accuracy > accept_acc:
        GB.fit(X,Y)
        pickle.dump(GB, open(model_s, 'wb'))
        message = "Success"
    else:
        print('Advising to use another model as the accuracy for this one is', accuracy)
        message = "Advising to use another model as the accuracy for this one is low"
        model_s = ""
    conf_matrix = confusion_matrix(Y_test, Y_pred)
    classi_report = classification_report(Y_test, Y_pred)
    print(conf_matrix)
    print(classi_report)    
    model_s1 = model_s    
    return accuracy, conf_matrix, classi_report, message, model_s1      

# Function for Support Vector Classifier


def SVC(X_train, Y_train, X_test, Y_test, model_xml, model_s, accept_acc, X, Y,df_train):
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.svm import SVC
    tree1 = ET.parse(model_xml)
    root1 = tree1.getroot()
    reg = root1[0].text
    ker = root1[1].text
    state = root1[2].text
    svc = SVC()
    parameters = svc.get_params()
    if reg != ' ':
        reg = float(reg)
    else:
        reg = parameters['C']
    if ker != ' ':
        ker = str(ker)
    else:
        ker = parameters['kernel']
    if state != ' ':
        state = int(state)
    else:
        state = parameters['random_state']
    svc = SVC(C = reg, kernel = ker, random_state = state)
    clf = OneVsRestClassifier(svc).fit(X_train, Y_train)
    pickle.dump(clf, open(model_s, 'wb'))
    imp_feat(df_train,svc)
    Y_pred = clf.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    print('SV Classifier accuracy: %0.3f'% accuracy)
    print("Sv Classifier report \n", classification_report(Y_test, Y_pred))
    if accuracy > accept_acc:
        clf.fit(X,Y)
        pickle.dump(clf, open(model_s, 'wb'))
        message = "Success"
    else:
        print('Advising to use another model as the accuracy for this one is', accuracy) 
        message = "Advising to use another model as the accuracy for this one is low"
        model_s = ""
    conf_matrix = confusion_matrix(Y_test, Y_pred)
    classi_report = classification_report(Y_test, Y_pred)
    print(conf_matrix)
    print(classi_report)
    model_s1 = model_s
    return accuracy, conf_matrix, classi_report, message, model_s1    

# Function for KNN Classifier


def KNNClassifier(X_train, Y_train, X_test, Y_test, model_xml, model_s, accept_acc, X, Y,df_train):
    from sklearn.neighbors import KNeighborsClassifier
    tree1 = ET.parse(model_xml)
    root1 = tree1.getroot()
    neighbors = root1[0].text
    w = root1[1].text
    KNN = KNeighborsClassifier()
    parameters = KNN.get_params()
    if neighbors != ' ':
        neighbors = int(neighbors)
    else:
        neighbors = parameters['n_neighbors']
    if w != ' ':
        w = int(w)
    else:
        w = parameters['weights']
    knn = KNeighborsClassifier(n_neighbors = neighbors, weights = w)
    knn.fit(X_train, Y_train)
    pickle.dump(knn, open(model_s, 'wb'))
    imp_feat(df_train,knn)
    Y_pred = knn.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    print('KNN Classifier accuracy: %0.3f'% accuracy)
    print("KNN Classifier report \n", classification_report(Y_test, Y_pred))
    if accuracy > accept_acc:
        knn.fit(X,Y)
        pickle.dump(knn, open(model_s, 'wb'))
        message = "Success"
    else:
        print('Advising to use another model as the accuracy for this one is', accuracy)
        message = "Advising to use another model as the accuracy for this one is low"
        model_s = ""
    conf_matrix = confusion_matrix(Y_test, Y_pred)
    classi_report = classification_report(Y_test, Y_pred)
    print(conf_matrix)
    print(classi_report) 
    model_s1 = model_s
    return accuracy, conf_matrix, classi_report, message, model_s1

# Function for KNN Regressor


def KNNRegressor(X_train, Y_train, X_test, Y_test, model_xml, model_s, accept_acc, X, Y,df_train):
    from sklearn.neighbors import KNeighborsRegressor
    tree1 = ET.parse(model_xml)
    root1 = tree1.getroot()
    neighbors = root1[0].text
    w = root1[1].text
    KNN = KNeighborsRegressor()
    parameters = KNN.get_params()
    if neighbors != ' ':
        neighbors = int(neighbors)
    else:
        neighbors = parameters['n_neighbors']
    if w != ' ':
        w = int(w)
    else:
        w = parameters['weights']
    knn = KNeighborsRegressor(n_neighbors = neighbors, weights = w)
    knn.fit(X_train, Y_train)
    pickle.dump(knn, open(model_s, 'wb'))
    imp_feat(df_train,knn)
    Y_pred = knn.predict(X_test)
    accuracy = r2_score(Y_test, Y_pred)
    print('KNN Regressor accuracy: %0.3f'% accuracy)
    if accuracy > accept_acc:
        knn.fit(X,Y)
        pickle.dump(knn, open(model_s, 'wb'))
        message = "Success"
    else:
        print('Advising to use another model as the accuracy for this one is', accuracy)
        message = "Advising to use another model as the accuracy for this one is low"
    conf_matrix = []
    classi_report = ""
    classi_report = str(classi_report)
    return accuracy, conf_matrix, classi_report, message, model_s1    
    

# Function for Naive Bayes Classifier

def NB(X_train, Y_train, X_test, Y_test, model_xml, model_s, accept_acc, X, Y,df_train):
    from sklearn.preprocessing import StandardScaler
    from sklearn.naive_bayes import GaussianNB
    #tree1 = ET.parse(model_xml)
    #root1 = tree1.getroot()
    #lr = root1[0].text
    #nest = root1[1].text
    #state = root1[2].text
    NB = GaussianNB()
    #parameters = NB.get_params()
    #if type(lr) != str:
        #lr = float(lr)
    #else:
        #lr = parameters['learning_rate']
    #if type(nest) != str:
        #nest = int(nest)
    #else:
        #nest = parameters['n_estimators']
    #if type(state) != str:
        #state = int(state)
    #else:
        #state = parameters['random_state']
    NB.fit(X_train, Y_train)
    pickle.dump(NB, open(model_s, 'wb'))
    imp_feat2(df_train,NB)
    Y_pred = NB.predict(X_test)  
    accuracy = accuracy_score(Y_test, Y_pred)
    print('NB Classifier accuracy: %0.3f'% accuracy)
    print("NB Classifier report \n", classification_report(Y_test, Y_pred))
    if accuracy > accept_acc:
        NB.fit(X,Y)
        pickle.dump(NB, open(model_s, 'wb'))
        message = "Success"
    else:
        print('Advising to use another model as the accuracy for this one is', accuracy)
        message = "Advising to use another model as the accuracy for this one is low"
        model_s = ""
    conf_matrix = confusion_matrix(Y_test, Y_pred)
    classi_report = classification_report(Y_test, Y_pred)
    print(conf_matrix)
    print(classi_report)
    model_s1 = model_s
    return accuracy, conf_matrix, classi_report, message, model_s1

# Function for XGB Classifier

def XGBC(X_train, Y_train, X_test, Y_test, model_xml, model_s, accept_acc, X, Y,df_train):
    import xgboost
    tree1 = ET.parse(model_xml)
    root1 = tree1.getroot()
    boost = root1[0].text
    maxdep = root1[1].text
    lr = root1[2].text
    tree = root1[3].text
    l2reg = root1[4].text
    l1reg = root1[5].text
    njobs = root1[6].text
    XGB = XGBClassifier()
    parameters = XGB.get_xgb_params()
    if boost != ' ':
        boost = str(boost)
    else:
        boost = parameters['booster']
    if maxdep != ' ':
        maxdep = int(maxdep)
    else: 
        maxdep = parameters['max_depth']
    if lr != ' ':
        lr = float(lr)
    else:
        lr = parameters['learning_rate']
    if tree != ' ':
        tree = str(tree)
    else: 
        tree = parameters['tree_method']
    if l2reg != ' ':
        l2reg = float(l2reg)
    else:
        l2reg = parameters['reg_lambda']
    if l1reg != ' ':
        l1reg = float(l1reg)
    else:
        l1reg = parameters['reg_alpha']
    if njobs != ' ':
        njobs = int(njobs)
    else:
        njobs = parameters['n_jobs']
    XGB = XGBClassifier(booster = boost, max_depth = maxdep, learning_rate = lr, tree_method = tree, reg_lambda = l2reg, reg_alpha = l1reg, n_jobs = njobs)  
    X_train = X_train.sample
    XGB.fit(X_train, Y_train)
    pickle.dump(XGB, open(model_s, 'wb'))
    imp_feat(df_train,XGB)
    Y_pred = XGB.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    print('XGB Classifier accuracy: %0.3f'% accuracy)
    print("XGB Classifier report \n", classification_report(Y_test, Y_pred))
    if accuracy > accept_acc:
        XGB.fit(X,Y)
        pickle.dump(XGB, open(model_s, 'wb'))
        message = "success" 
    else:
        print('Advising to use another model as the accuracy for this one is', accuracy)
        message = "Advising to use another model as the accuracy for this one is low"
        model_s = ""
    conf_matrix = confusion_matrix(Y_test, Y_pred)
    classi_report = classification_report(Y_test, Y_pred)
    print(conf_matrix)
    print(classi_report) 
    model_s1 = model_s
    return accuracy, conf_matrix, classi_report, message, model_s1   
        

# Function for Extra Trees Classifier


def ETC(X_train, Y_train, X_test, Y_test, model_xml, model_s, accept_acc, X, Y,df_train):
    from sklearn.ensemble import ExtraTreesClassifier
    tree1 =ET.parse(model_xml)
    root1 = tree1.getroot()
    nest = root1[0].text
    maxdep = root1[1].text
    maxfeat = root1[2].text
    njobs = root1[3].text
    state = root1[4].text
    ETC = ExtraTreesClassifier()
    parameters = ETC.get_params()
    if nest != ' ':
        nest = int(nest)
    else:
        nest = parameters['n_estimators']
    if maxdep != ' ':
        maxdep = int(maxdep)
    else:
        maxdep = parameters['max_depth']
    if maxfeat != ' ':
        maxfeat = int(maxfeat)
    else:
        maxfeat = parameters['max_features']
    if njobs != ' ':
        njobs = int(njobs)
    else:
        njobs = parameters['n_jobs']
    if state != ' ':
        state = int(state)
    else:
        state = parameters['random_state']
    ETC = ExtraTreesClassifier(n_estimators = nest, max_depth = maxdep, max_features = maxfeat, n_jobs = njobs, random_state = state)
    ETC.fit(X_train, Y_train)
    pickle.dump(ETC, open(model_s, 'wb'))
    imp_feat(df_train, ETC)
    Y_pred = ETC.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    print('Extra Trees Classifier accuracy: %0.3f'% accuracy)
    print("Extra Tress Classifier report \n", classification_report(Y_test, Y_pred))
    if accuracy > accept_acc:
        ETC.fit(X,Y)
        pickle.dump(ETC, open(model_s, 'wb'))
        message = "Success"
    else:
        print('Advising to use another model as the accuracy for this one is', accuracy)
        message = "Advising to use another model as the accuracy for this one is low"
        model_s = ""
    conf_matrix = confusion_matrix(Y_test, Y_pred)
    classi_report = classification_report(Y_test, Y_pred)
    print(conf_matrix)
    print(classi_report)
    model_s1 = model_s
    return accuracy, conf_matrix, classi_report, message, model_s1  


# Function for Decision Tree Classifier

def DTC(X_train, Y_train, X_test, Y_test, model_xml, model_s, accept_acc, X, Y,df_train):
    from sklearn.tree import DecisionTreeClassifier
    tree1 = ET.parse(model_xml)
    root1 = tree1.getroot()
    maxdep = root1[0].text
    maxfeat = root1[1].text
    state = root1[2].text
    DT = DecisionTreeClassifier()
    parameters = DT.get_params()
    if maxdep != ' ':
        maxdep = int(maxdep)
    else:
        maxdep = parameters['max_depth']
    if maxfeat !=' ':
        maxfeat = int(maxfeat)
    else:
        maxfeat = parameters['max_features']
    if state != ' ':
        state = int(state)
    else:
        state = parameters['random_state']
    DT = DecisionTreeClassifier(max_depth = maxdep, max_features = maxfeat, random_state = state)
    DT.fit(X_train, Y_train)
    pickle.dump(DT, open(model_s, 'wb'))
    imp_feat(df_train,DT)
    Y_pred = DT.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    print('Decision Tree Classification accuracy: %0.3f'% accuracy)
    print("Classifier report \n", classification_report(Y_test, Y_pred))
    if accuracy > accept_acc:
        DT.fit(X,Y)
        pickle.dump(DT, open(model_s, 'wb'))
        message = "Success"
    else:
        print('Advising to use another model as the accuracy for this one is', accuracy)
        message = "Advising to use another model as the accuracy for this one is low"
        model_s = ""
    conf_matrix = confusion_matrix(Y_test, Y_pred)
    classi_report = classification_report(Y_test, Y_pred)
    print(conf_matrix)
    print(classi_report)  
    model_s1 = model_s
    return accuracy, conf_matrix, classi_report, message, model_s1  

# Function for Polynomial Linear Regression


def PLR(X_train, Y_train, X_test, Y_test, model_xml, model_s, accept_acc, X, Y,df_train):
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    tree1 = ET.parse(model_xml)
    root1 = tree1.getroot()
    degree = root1[0].text
    PLR = LinearRegression()
    parameters = PLR.get_params()
    print(parameters)
    poly_reg = PolynomialFeatures(degree = degree)
    X_poly = poly_reg.fit_transform(X_train)
    
    PLR.fit(X_poly, Y_train)
    pickle.dump(PLR, open(model_s, 'wb'))
    
    imp_feat2(df_train,PLR)
    Y_pred = PLR.predict(X_test)
    accuracy = r2_score(Y_test, Y_pred)
    print('Polynomial regressor r2 score: %0.3f'% accuracy)
    if accuracy > accept_acc:
        PLR.fit(X,Y)
        pickle.dump(PLR, open(model_s, 'wb'))
        message = "Success"
    else:
        print('Advising to use another model as the score for this one is', accuracy)
        message = "Advising to use another model as the score for this one is low"
        model_s = ""
    conf_matrix = []
    summ = ""
    summ = str(summ)
    model_s1 = model_s
    return accuracy, conf_matrix, summ, message, model_s1    

# Function for Decision Tree Regression


def DTR(X_train, Y_train, X_test, Y_test, model_xml, model_s, accept_acc, X, Y,df_train):
    from sklearn.tree import DecisionTreeRegressor
    tree1 = ET.parse(model_xml)
    root1 = tree1.getroot()
    maxdep = root1[0].text
    maxfeat = root1[1].text
    state = root1[2].text
    DT = DecisionTreeRegressor()
    parameters = DT.get_params()
    if maxdep != ' ':
        maxdep = int(maxdep)
    else:
        maxdep = parameters['max_depth']
    if maxfeat != ' ':
        maxfeat = int(maxfeat)
    else:
        maxfeat = parameters['max_features']
    if state != ' ':
        state = int(state)
    else:
        state = parameters['random_state']
    DT = DecisionTreeRegressor(max_depth = maxdep, max_features = maxfeat, random_state = state)
    DT.fit(X_train, Y_train)
    pickle.dump(DT, open(model_s, 'wb'))
    imp_feat(df_train,DT)
    Y_pred = DT.predict(X_test)
    accuracy = r2_score(Y_test, Y_pred)
    print('Decision tree regressor r2 score: %0.3f'% accuracy)
    if accuracy > accept_acc:
        DT.fit(X,Y)
        pickle.dump(DT, open(model_s, 'wb'))
        message = "Success"
    else:
        print('Advising to use another model as the score for this one is', accuracy)
        message = "Advising to use another model as the score for this one is low"
        model_s = ""
    conf_matrix = []
    summ = ""
    summ = str(summ)
    print(summ)    
    model_s1 = model_s
    return accuracy, conf_matrix, summ, message, model_s1

# Function for Random Forest Regressor

def RFR(X_train, Y_train, X_test, Y_test, model_xml, model_s, accept_acc, X, Y,df_train):
    from sklearn.ensemble import RandomForestRegressor
    tree1 = ET.parse(model_xml)
    root1 = tree1.getroot()
    nest = root1[0].text
    state = root1[1].text
    njobs = root1[2].text
    RF = RandomForestRegressor()
    parameters = RF.get_params()
    if nest != ' ':
        nest = int(nest)
    else:
        nest = parameters['n_estimators']
    if state != ' ':
        state = int(state)
    else:
        state = parameters['random_state']
    if njobs != ' ':
        njobs = int(njobs)
    else:
        njobs = parameters['n_jobs']
    RF = RandomForestRegressor(n_estimators = nest, random_state = state, n_jobs = njobs)
    RF.fit(X_train, Y_train)
    pickle.dump(RF, open(model_s, 'wb'))
    imp_feat(df_train,RF) # move under accuracy > accept_acc: after testing
    Y_pred = RF.predict(X_test)
    accuracy = r2_score(Y_test, Y_pred)
    print('Random Forest regressor r2 score: %0.3f'% accuracy)
    if accuracy > accept_acc:
        RF.fit(X,Y)
        pickle.dump(RF, open(model_s, 'wb'))
        message = "Success"
    else:
        print('Advising to use another model as the score for this one is', accuracy)
        message = "Advising to use another model as the score for this one is low"
        model_s = ""
    conf_matrix = [] # will be empty in the regressors
    
    summ = "" # check how summary can be taken for this algo
    summ = str(summ)
    print(summ)
    model_s1 = model_s
    return accuracy, conf_matrix, summ, message, model_s1   

# Function for Gradient Boost Regressor

def GBR(X_train, Y_train, X_test, Y_test, model_xml, model_s, accept_acc, X, Y,df_train):
    from sklearn.ensemble import GradientBoostingRegressor
    tree1 = ET.parse(model_xml)
    root1 = tree1.getroot()
    lr = root1[0].text
    nest = root1[1].text
    state = root1[2].text
    GB = GradientBoostingRegressor()
    parameters = GB.get_params()
    if lr != ' ':
        lr = float(lr)
    else:
        lr = parameters['learning_rate']
    if nest != ' ':
        nest = int(nest)
    else:
        nest = parameters['n_estimators']
    if state != ' ':
        state = int(state)
    else:
        state = parameters['random_state']
    GB = GradientBoostingRegressor(learning_rate = lr, n_estimators = nest, random_state = state)
    GB.fit(X_train, Y_train)
    pickle.dump(GB, open(model_s, 'wb'))
    imp_feat(df_train, GB)
    Y_pred = GB.predict(X_test)
    accuracy = r2_score(Y_test, Y_pred)
    print('Gradient boosting Regressor r2 score: %0.3f'% accuracy)
    if accuracy > accept_acc:
        GB.fit(X,Y)
        pickle.dump(GB, open(model_s, 'wb'))
        message = "Success"
    else:
        print('Advising to use another model as the score for this one is', accuracy)
        message = "Advising to use another model as the score for this one is low"
        model_s = ""
    conf_matrix = []
    summ = ""
    summ = str(summ)
    model_s1 = model_s
    return accuracy, conf_matrix ,summ , message, model_s1    

# Function for Multiple Linear Regression

def MLR(X_train, Y_train, X_test, Y_test, model_xml, model_s, accept_acc, X, Y,df_train):
    from sklearn.linear_model import LinearRegression
    tree1 = ET.parse(model_xml)
    root1 = tree1.getroot()
    maxdep = root1[0].text
    maxfeat = root1[1].text
    state = root1[2].text
    MLR = LinearRegression()
    parameters = MLR.get_params()
    #if type(maxdep) != str:
        #maxdep = int(maxdep)
    #else:
        #maxdep = parameters['max_depth']
    #if type(maxfeat) !=str:
        #maxfeat = int(maxfeat)
    #else:
        #maxfeat = parameters['max_features']
    #if type(state) != str:
        #state = int(state)
    #else:
        #state = parameters['random_state']
    #MLR = LinearRegression(max_depth = maxdep, max_features = maxfeat, random_state = state)
    MLR.fit(X_train, Y_train)
    pickle.dump(MLR, open(model_s, 'wb'))
    imp_feat2(df_train,MLR)
    Y_pred = MLR.predict(X_test)
    accuracy = r2_score(Y_test, Y_pred)
    print('Multi Linear regressor r2 score: %0.3f'% accuracy)
    if accuracy > accept_acc:
        MLR.fit(X,Y)
        pickle.dump(MLR, open(model_s, 'wb'))
        message = "Success"
    else:
        print('Advising to use another model as the score for this one is', accuracy)
        message = "Advising to use another model as the score for this one is low"
    conf_matrix = []
    # Add a constant to get an intercept
    X_train_sm = sm.add_constant(X_train)

    # Fit the resgression line using 'OLS'
    lr = sm.OLS(Y_train, X_train_sm).fit()   
    summ = lr.summary()
    summ = str(summ)
    print(summ)
    print(model_s)
    model_s1 = model_s
    return accuracy, conf_matrix ,summ , message , model_s1

# Function for AR Time Series 


def AR(model_xml,X,df_train):
    from statsmodels.tsa.ar_model import AutoReg
    
    tree1 = ET.parse(model_xml)
    root1 = tree1.getroot()
    prediction_period = root1[0].text
    n_lags = root1[1].text
    
    #prediction_period = 7 # read from AR.xml
    #n_lags = 2 # read from AR.xml
    #df_train = pd.read_csv(train_data,index_col='Date',parse_dates=True)
    # split dataset
    X = df_train.values
    print(X)
    #print(X.head())
    print(len(X))
    train, test = X[1:len(X) - prediction_period], X[len(X) - prediction_period:] 
    print(train)
    print(test)
    # train autoregression
    print(n_lags)
    model = AutoReg(train, lags= n_lags)
    model_fit = model.fit()
    print('Coefficients: %s' % model_fit.params)
    # make predictions
    predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
    for i in range(len(predictions)):
     	print('predicted=%f, expected=%f' % (predictions[i], test[i]))
    rmse = sqrt(mean_squared_error(test, predictions))
    print('Test RMSE: %.3f' % rmse)
    # plot results
    pyplot.plot(test)
    pyplot.plot(predictions, color='red')
    pyplot.show()
    return rmse



def SEQ(in_dim, out_dim,model_xml):
    from keras.models import Sequential
    from keras.layers import Dense, Activation, Dropout
    
    tree1 = ET.parse(model_xml)
    root1 = tree1.getroot()
    u = root1[0].text
    initial = root1[1].text
    activation_hid = root1[2].text
    activation_out = root1[3].text
    l = root1[4].text
    opt = root1[5].text
    met = root1[6].text
    drop = root1[7].text
    ep = root1[8].text
    batch = root1[9].text
    verb = root1[10].text
    if u != ' ':
        u = int(u)
    else:
        u = 16
    if initial != ' ':
        initial = str(initial)
    else: 
        initial = 'uniform'
    if activation_hid != ' ':
        activation_hid = str(activation_hid)
    else:
        activation_hid = 'relu'    
    if activation_out != ' ':
        activation_out = str(activation_out)
    else:
        activation_out = 'softmax'
    if l != ' ':
        l = str(l)
    else:
        l = 'binary_crossentropy'
    if opt != ' ':
        opt = str(opt)
    else:
        opt = 'adam'
    if met != ' ':
        met = str(met)
    else:
        met = ['accuracy']
    if drop != ' ':
        drop = float(drop)
    else:
        drop = 0.5
    if ep != ' ':
        ep = int(ep)
    else:
        ep = 20
    if batch != ' ':
        batch = int(batch)
    else:
        batch = 1024
    if verb != ' ':
        verb = int(verb)
    else:
        verb = 1
    model = Sequential()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = test_ratio, random_state = state)
    model.add(Dense(u, init = initial, input_dim = in_dim, activation = activation_hid))
    model.add(Dropout(drop))
    
    model.add(Dense(u, init = initial, activation = activation_hid))
    model.add(Dropout(drop))
    
    model.add(Dense(out_dim, init = initial, activation = activation_out))
     
    model.compile(loss = l, optimizer = opt, metrics = met)
    
    model.fit(X_train, Y_train, epochs = ep, batch_size = batch, verbose = verb)
    model.save_weights("model.h5")
    Y_pred = model.predict(X_test)
    #accuracy = model.evaluate(X_test, Y_test)
    #accuracy = np.mean(accuracy)
    accuracy = accuracy_score(Y_pred, Y_test)
    print('Keras Classifier accuracy: %0.3f'% accuracy)
    if accuracy > accept_acc:
        model.fit(X_train, Y_train, epochs = ep, batch_size = batch, verbose = verb)
        model.save_weights("model.h5")
        message = "Success"
    else:
        print('Advising to use another model as the accuracy for this one is', accuracy)
        message = "Advising to use another model as the accuracy for this one is low"
    return accuracy    
    


def readandtrain(train_data,drop_col,output_col,test_r,st,samp,model_n,accept_acc,model_s, model_xml):
    accuracy = 0

    # store training in a Dataframe
    df_train = pd.read_csv(train_data)
    print('Read training data into a Pandas DataFrame')
    
    # Creating Input dataframe X and Output dataframe Y
    if ',' in drop_col:
        drop_col = drop_col.split(",")
    if drop_col == ' ':
        X = df_train.drop(columns = [output_col], axis = 1)
        Y = df_train[output_col]
    elif output_col == ' ':
        X = df_train.drop(drop_col, axis = 1)
    else:
        X = df_train.drop(columns = [drop_col, output_col], axis = 1)
        Y = df_train[output_col]
    print('Successfully created Input & Output dataframes')
    
    # Splitting Dataset into Train and Test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = test_r, random_state = st)
    print('Successfylly splitted the dataset into training and testing sets')
    
    # Sampling Dataset if mentioned 
    if samp == 'over':
        class_count(df_train, output_col)
        oversampler = SMOTE(random_state = 0)
        X_train, Y_train = oversampler.fit_sample(X_train, Y_train)
        plt.figure()
        Series(Y_train).value_counts().sort_index().plot(kind = 'bar')
        plt.ylabel("Count")
        plt.title('ARTIFICIAL CLASS COUNT')
    elif samp == 'under':
        class_count(df_train, output_col)
        undersampler = RandomUnderSampler(random_state = 0)
        X_train, Y_train = undersampler.fit_sample(X_train, Y_train)
        plt.figure()
        Series(Y_train).value_counts().sort_index().plot(kind = 'bar')
        plt.ylabel("Count")
        plt.title('ARTIFICIAL CLASS COUNT')
    print('Sampling function executed')
        
        
    if model_n == 'LogisticRegression':
        accuracy, conf_matrix, classi_report, message, model_s1 = LOR(X_train, Y_train, X_test, Y_test, model_xml, model_s, accept_acc, X, Y, df_train)
    elif model_n == 'RandomForestClassifier':
        accuracy, conf_matrix, classi_report, message, model_s1 = RFC(X_train, Y_train, X_test, Y_test, model_xml, model_s, accept_acc, X, Y, df_train)
    elif model_n == 'LinearRegression':
        accuracy, conf_matrix, classi_report, message, model_s1 = LR(X_train, Y_train, X_test, Y_test, model_xml, model_s, accept_acc, X, Y, df_train)    
    elif model_n == 'AdaBoostClassifier':
        accuracy, conf_matrix, classi_report, message, model_s1 = ABC(X_train, Y_train, X_test, Y_test, model_xml, model_s, accept_acc, X, Y, df_train)    
    elif model_n == 'GradientBoostingClassifier':
        accuracy, conf_matrix, classi_report, message, model_s1 = GBC(X_train, Y_train, X_test, Y_test, model_xml, model_s, accept_acc, X, Y, df_train)
    elif model_n == 'SupportVectorClassifier':
        accuracy, conf_matrix, classi_report, message, model_s1 = SVC(X_train, Y_train, X_test, Y_test, model_xml, model_s, accept_acc, X, Y, df_train)
    elif model_n == 'KNNClassifier':
        accuracy, conf_matrix, classi_report, message, model_s1 = KNNClassifier(X_train, Y_train, X_test, Y_test, model_xml, model_s, accept_acc, X, Y, df_train)
    elif model_n == 'NaiveBayes':
        accuracy, conf_matrix, classi_report, message, model_s1 = NB(X_train, Y_train, X_test, Y_test, model_xml, model_s, accept_acc, X, Y, df_train)
    elif model_n == 'XGBoostClassifier':
        accuracy, conf_matrix, classi_report, message, model_s1 = XGBC(X_train, Y_train, X_test, Y_test, model_xml, model_s, accept_acc, X, Y, df_train)    
    elif model_n == 'ExtraTreesClassifier':
        accuracy, conf_matrix, classi_report, message, model_s1 = ETC(X_train, Y_train, X_test, Y_test, model_xml, model_s, accept_acc, X, Y, df_train)
    elif model_n == 'DecisionTreeClassifier':
        accuracy, conf_matrix, classi_report, message, model_s1 = DTC(X_train, Y_train, X_test, Y_test, model_xml, model_s, accept_acc, X, Y, df_train)
    elif model_n == 'SupportVectorRegressor':
        accuracy, conf_matrix, classi_report, message, model_s1 = SVR(X_train, Y_train, X_test, Y_test, model_xml, model_s, accept_acc, X, Y, df_train)
    elif model_n == 'KNNRegressor':
        accuracy, conf_matrix, classi_report, message, model_s1 = KNNRegressor(X_train, Y_train, X_test, Y_test, model_xml, model_s, accept_acc, X, Y, df_train)   
    elif model_n == 'PolynomialRegression': 
        accuracy, conf_matrix, classi_report, message, model_s1 = PLR(X_train, Y_train, X_test, Y_test, model_xml, model_s, accept_acc, X, Y,df_train)
    elif model_n == 'DecisionTreeRegressor': 
        accuracy, conf_matrix, classi_report, message, model_s1 = DTR(X_train, Y_train, X_test, Y_test, model_xml, model_s, accept_acc, X, Y, df_train)
    elif model_n == 'RandomForestRegressor': 
        accuracy, conf_matrix, classi_report, message, model_s1 = RFR(X_train, Y_train, X_test, Y_test, model_xml, model_s, accept_acc, X, Y,df_train)
    elif model_n == 'GradientBoostingRegressor': 
        accuracy, conf_matrix, classi_report, message, model_s1 = GBR(X_train, Y_train, X_test, Y_test, model_xml, model_s, accept_acc, X, Y,df_train)
    elif model_n == 'MultipleLinearRegression': 
        accuracy, conf_matrix, classi_report, message, model_s1 = MLR(X_train, Y_train, X_test, Y_test, model_xml, model_s, accept_acc, X, Y, df_train)
    elif model_n == 'AR_Timeseries': 
        accuracy ==  AR(model_xml,X,df_train)   
    
    elif model_n == 'Keras':
        X = np.array(X)
        in_dim = X.shape[1]
        print("\nInput dim: ", in_dim)
        if type(Y) == np.array:
            out_dim = int(Y.shape[1])
        else:
            out_dim = 1
            print("\nOutput dim: ", out_dim)
            accuracy ==  SEQ(in_dim, out_dim,model_xml)    
    else:
        print('Enter a valid model_name')
        
    return accuracy, conf_matrix, classi_report, message, model_s1    
