# -*- coding: utf-8 -*-
"""

@author: Robert Kajnak
"""

import numpy as np
from numbers import Number

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


