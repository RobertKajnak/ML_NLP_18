# -*- coding: utf-8 -*-
"""

@author: Robert Kajnak
"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn_crfsuite.metrics import flat_classification_report
from hmmlearn import hmm as hmm_old
import nltk.tag.hmm as hmm
from nltk.probability import LidstoneProbDist as LidstoneProbDist
import sklearn_crfsuite


def gen_rep(data,y_pred,format_dict):
    return classification_report(data.y_test,y_pred,
                        labels=data.labels_num,target_names=data.labels_name,
                        output_dict = format_dict,digits = 3)    
def gen_rep_flat(data,y_pred,format_dict):
    return flat_classification_report(
                data.y_test, y_pred, digits=3,
                output_dict = format_dict)

#%%Naive Bayes
def NB(data,verbose=True):
    '''
    NB(data,verbose)->model,prediction,report(dict). 
    verbose=>prints report
    for data structure see preprocessing.py
    '''
    gnb = MultinomialNB(alpha=0.001,fit_prior=False)
    y_pred = gnb.fit(data.x_train, data.y_train).predict(data.x_test)
    
    #print("Number of mislabeled points out of a total %d points : %d" % (x_test.shape[0],(y_test != y_pred).sum()))
    
    if verbose:
        print("Naive Bayes results:")
        print(gen_rep(data,y_pred,False))

    return gnb,y_pred,gen_rep(data,y_pred,True)

#%%Logisic Regression
def LR(data,verbose=True):
    '''
    NB(data,verbose)->model,prediction,report(dict). 
    verbose=>prints report
    for data structure see preprocessing.py
    '''
    # using the parameters recommended on the sklearn documentaiton increased
    #the performance by 2%, but increases run time by 100 times:
    #random_state=0, solver='lbfgs',multi_class='ovr'
    
    clf = LogisticRegression(solver='lbfgs', multi_class='auto').fit(data.x_train, data.y_train)
    y_pred = clf.predict(data.x_test)
    
    if verbose:
        print("Logistic Regression results:")
        print(gen_rep(data,y_pred,False))
    #clf.score(x_test,y_test)
    return clf,y_pred,gen_rep(data,y_pred,True)

#%% SVM

def SVM(data,verbose=True):
    '''
    NB(data,verbose)->model,prediction,report(dict). 
    verbose=>prints report
    for data structure see preprocessing.py
    '''
    clf = svm.LinearSVC()
    clf.fit(data.x_train, data.y_train)  
    
    y_pred = clf.predict(data.x_test)
    
    if verbose:
        print("Support Vector Machine results:")
        print(gen_rep(data,y_pred,False))
    return clf,y_pred,gen_rep(data,y_pred,True)

#%% HMM
def HMM_old(data,verbose=True):
    '''Deprecated - will be removed in next version'''
    hm = hmm_old.GaussianHMM(n_components=2, n_iter=100)

        
    '''temp = transl.idx;
    transl.idx=10
    hm.fit(oneHot(y_train,transl))
    transl.idx=temp'''
    hm.fit(data.y_train.reshape(-1,1))
    #y_pred = hm.score(y_test.reshape(-1,1))
    y_pred = hm.predict(data.y_test.reshape(-1,1))

    if verbose:
        print("HMM results:")
        print(print(gen_rep,y_pred,False))
        
    return hm,y_pred,print(gen_rep,y_pred,True)

def HMM(data,symbols,tag_set,verbose=True):
    '''
    NB(data,symbols,tag_set,verbose)->model,prediction,report(dict). 
    Keyword arguments:
        data: see preprocessing.py
        symbols: list of the input class labels
        tag_set: list of the output class labels
    for data structure see preprocessing.py
    '''
    trainer = hmm.HiddenMarkovModelTrainer(tag_set, symbols)
    tagger = trainer.train_supervised(
        data.y_train,
        estimator=lambda fd, bins: LidstoneProbDist(fd, 0.1, bins),
    )

    y_pred = []
    for sentence in data.x_test:
        y_pred.append(tagger.tag(sentence))
    
    #unlike the test or evaluate function from the same suit, this requires 
    #a list of symbols, not tuples of symbols and tags
    y_pred = [[tup[1] for tup in sentence] for sentence in y_pred]

    print('HMM Results:')
    print(gen_rep_flat(data,y_pred,False))
    return tagger,y_pred,gen_rep_flat(data,y_pred,True)
#%% CRF
def CRF(data,verbose=True):
    '''
    NB(data,verbose)->model,prediction,report(dict). 
    verbose=>prints report
    for data structure see preprocessing.py
    '''
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
        print(gen_rep_flat(data,y_pred,False))
    return crf,y_pred,gen_rep_flat(data,y_pred,True)
    