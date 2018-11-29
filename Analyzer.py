# -*- coding: utf-8 -*-
"""

@author: Robert Kajnak
"""
from random import random
import numpy as np
from numbers import Number
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn_crfsuite.metrics import flat_classification_report
from hmmlearn import hmm
import sklearn_crfsuite

#%%Read the entries from the file and add them to a list
#with optional filtering
def readWords():
    f = open('reuters-train.en','r')
    
    words = []
    for line in f:
        l = line.replace('\n','').split(' ')
        if len(l)==1: #\
           #or l[0] in '!@#$%^&*()12346789[]{}\\|\'\",./<>?' \
            #:
            continue
        words.append(l)
    
    f.close()
    return words

#%% Generate new features based on words:
    
def appendFeatures(words):
    words_upgraded = []
    maxi = len(words)
    for i in range(maxi):
        word = words[i][0]
        words_upgraded.append([
                word,
                word.lower(),
                words[i][1],
                words[i][2],
                '_lower:'+ str(word.islower()),
                '_upper:'+ str(word.isupper()),
                '_digit:'+ str(word.isdigit()),
                '_title:'+ str(word.istitle()),
                '_x:' + str('x' in word),
                '_y:' + str('y' in word),
                '_length:' + str(len(word)),
                word[-2:],
                word[-3:],
                '-1:'+words[i-1][1] if i>0 else '-1:-',#POS
                '-1:'+words[i-1][2] if i>0 else '-1:-',#Segment
                '-1:'+words[i-1][0] if i>0 else '-1:-', #previous word
                #'-1:'+words[i-1][3] if i>1 else '-1:-',#label
                '+1:'+words[i+1][1] if i<maxi-1 else '+1:-',#POS
                '+1:'+words[i+1][2] if i<maxi-1 else '+1:-',#Segment
                '+1:'+words[i+1][0] if i<maxi-1 else '+1:-',#next word
                '+1:'+words[i+1][3] if i<maxi-1 else '+1:-',#label
                words[i][3] #the label, is only included in the Y, not the X
                ])
    
    return words_upgraded
#TODO add nr of capital letters
#%%Reshape to the desired one;
#doing this while reading it would increase speed, but this is better for modularity

class translator:
    def __init__(self):
        self.words={}
        self.idx = 0
    def translate(self,word):
        if isinstance(word,Number):
            for k,v in self.words.items():
                if v==word:
                    return k
            return -1
        else:
            if not (word in self.words):
                self.words[word]= self.idx
                self.idx+=1
                return self.idx-1
            else:
                return self.words[word]

#ensure all words are similarly split
                
def createDataset(words):
    
    #the 0 class seems to outweigh all other classes by more than 10 times, so
    #it will be reduced here
    new_words=[]
    for word in words:
        if word[-1] == 'O':
            #if random() < 0.06:
            new_words.append(word)
        else:
            new_words.append(word)
    words= new_words
    
    T = translator()
    
    nw = len(words)
    nf = len(words[0])-1
    
    #X = np.empty([nw,nf], dtype ='<U30') 
    #Y = np.empty([nw], dtype ='<U30')
    X = np.zeros([nw,nf])
    Y = np.zeros([nw])
    
    for i,word in enumerate(words):
        Y[i] = T.translate(word[-1])
    for j in range(nf):
        for i,word in enumerate(words):    
            X[i][j] = T.translate(words[i][j])
        
    
    labels_num = sorted(list(set(Y)))
    labels_name =[]
    for label in labels_num:
        labels_name.append(T.translate(label))

    return X,Y,T,labels_num,labels_name

#The one-hot encoding greatly increases performance
#Potential bug: if there is e.g. a word "B-LOC" in the corpus, it will be encoded
#the same way as the actual NE label
def oneHot(X, transl):
    from scipy.sparse import lil_matrix
    X_new = lil_matrix((len(X),transl.idx),dtype=np.int8)
    #X_new  = np.zeros([len(X),transl.idx])
    for i in range(X.shape[0]):
        if (len(X.shape)==1):
            X_new[i,int(X[i])] = 1
        else:   
            for j in range(X.shape[1]):
                X_new[i,int(X[i][j])] = 1
    return X_new


def shuffle_parallel(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    
class data_wrap:
    def __init__(self,x_train,y_train,x_test,y_test,transl=None,labels_num=None,labels_name=None):
        self.x_train=x_train
        self.y_train=y_train
        self.x_test=x_test
        self.y_test = y_test
        self.trans = transl
        self.labels_num = labels_num
        self.labels_name = labels_name
#%%Preparation for using ML algorithms

#Split training and test sets
def split_tr(X,Y,ratio):
    lim = (np.int)(len(Y)*ratio)
    return X[:lim],Y[:lim],X[lim:],Y[lim:]



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
    
    clf = LogisticRegression().fit(data.x_train, data.y_train)
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

#%%Construct dictionary for CRF
def words2dictionary(words,feature_names=None):
    new_tokens = []
    new_labels = []
    if feature_names==None:
        feature_names=[]
        for i in range(len(words[0])-1):
            feature_names.append(str(i))
            
    token_sentence = []
    label_sentence = []
    for word in words:
        new_word = {}
        for idx,feature in enumerate(feature_names):
            new_word[feature] = word[idx]
        token_sentence.append(new_word)
        label_sentence.append(word[idx+1])
        if word[1]=='.':
            new_tokens.append(token_sentence)
            token_sentence = []
            new_labels.append(label_sentence)
            label_sentence = []
        #new_words.append(new_word)
        
    return new_tokens, new_labels


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
    
#%%Handy method for testing the models
'''
    Runs all the models that are specified with the specified word set
    PARAMS:
        WORDS: list of list of words and features. 
            Format: n*m. n=nr of words, m=nr features + expected output (single)
        MODELS: a string containing the model names. Order is not important.
            Possible models are: NB, LR, SVM, HMM, CRF. Coming soon: CNN
            If a model is specified twice, it will be run twice. The input is
            randomized only once, where applicable
        VERBOSE: 0: print nothing
                1: print results
                2: print status messages:
                3: print both
    RETURN: 
        returns a list containing the output vectors and a list of models 
        of the sepcified inputs, in the order where they were provided
'''
def runModels(words, models, verbose):
    # Preparing data for one-hot encodign -- converts strings into integers
    if any(i in models for i in ['HMM','NB','LR','SVM']):
        verbose|2 and print('Initial pre-processing...')
        X,Y,transl,labels_num,labels_name = createDataset(words)
        
    # Algoirthms using non-randomized, one-hot data:HMM
    if 'HMM' in models:
        verbose|2 and print('Preprocessing data for HMM...')
        X_onehot_ord = oneHot(X,transl)
        x_train_oh,y_train_oh,x_test_oh,y_test_oh = split_tr(X_onehot_ord,Y,0.8)
        data_ordered_oh = data_wrap(x_train_oh,y_train_oh,x_test_oh,y_test_oh,transl,labels_num,labels_name)
    
    # Algorithms using shuffled, one-hot data:NB,LR,SVM
    if any(i in models for i in ['NB','LR','SVM']):
        verbose|2 and print('Preprocessing data for NB, LR and/or SVM...')
        shuffle_parallel(X,Y)
        X_onehot_sh = oneHot(X,transl)
        x_train_oh_sh,y_train_oh_sh,x_test_oh_sh,y_test_oh_sh = split_tr(X_onehot_sh,Y,0.8)
        data_shuffled = data_wrap(x_train_oh_sh,y_train_oh_sh,x_test_oh_sh,y_test_oh_sh,transl,labels_num,labels_name)

    #Ordered, using sentences: CRF
    if 'CRF' in models:
        verbose|2 and print('Preprocessing data for CRF...')
        tokens_dict,labels_dict = words2dictionary(words)#,['token','POS','segment'])
        
        tokens_train,labels_train,tokens_test,labels_test = split_tr(tokens_dict,labels_dict,0.8)
        data_dictionary = data_wrap(tokens_train,labels_train,tokens_test,labels_test)
    
    model_results = []
    model_objects = []
    def _add_to_output(model_y_pred):
        model_objects.append(model_y_pred[0])
        model_results.append(model_y_pred[1])
        
    for model in models:
        if 'HMM' in model:
            verbose|2 and print('Running HMM...')
            _add_to_output(HMM(data_ordered_oh,verbose|1))
            
        if 'NB' in model:
            verbose|2 and print('Running NB...')
            _add_to_output( NB(data_shuffled,verbose|1))
            
        if 'LR' in model:
            verbose|2 and print('Running LR...')
            _add_to_output(LR(data_shuffled,verbose|1))
            
        if 'SVM' in model:
            verbose|2 and print('Running SVM...')
            _add_to_output(SVM(data_shuffled,verbose|1))
            
        if  'CRF' in model:
            verbose|2 and print('Running CRF...')
            _add_to_output(CRF(data_dictionary,verbose|1))
            
    return model_objects,model_results
            
#%% Main
if __name__ == "__main__":
    words = readWords()
    words = appendFeatures(words)
    
    models=[
            'NB',
            'LR',
            'SVM', 
            'HMM',
            'CRF'
            ]
    runModels(words, models, verbose = 3)
    