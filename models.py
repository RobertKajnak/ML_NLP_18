# -*- coding: utf-8 -*-
"""

@author: Robert Kajnak
"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn_crfsuite.metrics import flat_classification_report
from hmmlearn import hmm
import sklearn_crfsuite



#%%Naive Bayes
def NB(data,verbose=True):
    gnb = MultinomialNB(alpha=0.001,fit_prior=False)
    y_pred = gnb.fit(data.x_train, data.y_train).predict(data.x_test)
    
    #print("Number of mislabeled points out of a total %d points : %d" % (x_test.shape[0],(y_test != y_pred).sum()))
    
    if verbose:
        print("Naive Bayes results:")
        print(classification_report(data.y_test,y_pred,labels=data.labels_num,target_names=data.labels_name))

    return gnb,y_pred

#%%Logisic Regression
def LR(data,verbose=True):
    # using the parameters recommended on the sklearn documentaiton increased
    #the performance by 2%, but increases run time by 100 times:
    #random_state=0, solver='lbfgs',multi_class='ovr'
    
    clf = LogisticRegression(solver='lbfgs', multi_class='auto').fit(data.x_train, data.y_train)
    y_pred = clf.predict(data.x_test)
    
    if verbose:
        print("Logistic Regression results:")
        print(classification_report(data.y_test,y_pred,labels=data.labels_num,target_names=data.labels_name))
    #clf.score(x_test,y_test)
    return clf,y_pred

#%% SVM

def SVM(data,verbose=True):
    clf = svm.LinearSVC()
    clf.fit(data.x_train, data.y_train)  
    
    y_pred = clf.predict(data.x_test)
    if verbose:
        print("Support Vector Machine results:")
        print(classification_report(data.y_test,y_pred,labels=data.labels_num,target_names=data.labels_name))
    return clf,y_pred

#%%HMM
def HMM(data,verbose=True):
    hm = hmm.GaussianHMM(n_components=2, n_iter=100)

        
    '''temp = transl.idx;
    transl.idx=10
    hm.fit(oneHot(y_train,transl))
    transl.idx=temp'''
    hm.fit(data.y_train.reshape(-1,1))
    #y_pred = hm.score(y_test.reshape(-1,1))
    y_pred = hm.predict(data.y_test.reshape(-1,1))

    if verbose:
        print("HMM results:")
        print(classification_report(data.y_test,y_pred,labels=data.labels_num,target_names=data.labels_name))
        
    return hm,y_pred


#%%CRF
def CRF(data,verbose=True):
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=False
    )
    crf.fit(data.x_train, data.y_train)    
    y_pred = crf.predict(data.x_test)
    
    if verbose:
        print("CRF results:")
        print(flat_classification_report(
                data.y_test, y_pred, digits=3))
    return crf,y_pred
    