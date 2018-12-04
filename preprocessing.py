# -*- coding: utf-8 -*-
"""

@author: Robert Kajnak
"""

import numpy as np
from numbers import Number
from geotext import GeoText
import random
from functools import reduce

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

def word_shape(word):
    #convert capital letters into X, non-capitals into x, digits into d and punctuation unchanged
    shape = list(map(lambda c: 'x' if c.islower() else 'X' if c.isupper() else 'd' if c.isdigit() else c, word))
    # convert list into string
    shape = reduce(lambda s,c: s+c,shape)
    #keep first and last two characters and the type of characters between them
    #e.g. 'xxXXxd-:XX' -> xxxDXd-:XX'
    #the order is a pre-determined one. If order should be kept, check the commented reduction
    shape = shape[:2] + \
                ('x' if 'x' in shape[2:-2] else '') + \
                ('X' if 'X' in shape[2:-2] else '') + \
                ('d' if 'd' in shape[2:-2] else '') + \
                shape[2:-2].replace('x','').replace('X','').replace('d','')+\
                shape[-2:]
    '''        shape = shape[:2] + \
                reduce((lambda s,c: s + \
                        ('x' if ('x' in c and 'x' not in s) else 
                       'X' if ('X' in c and 'X' not in s) else 
                       'd' if ('d' in c and 'd' not in s) else
                       c if (c not in s) else '')), shape[2:-2]) + \
                shape[-2:]'''
    return shape
            

def appendFeatures(words):
    words_upgraded = []
    maxi = len(words)
    for i in range(maxi):
        word = words[i][0]
        geo = GeoText(word)
        
        #.replace() would be slower
        
                
        words_upgraded.append([
                word,
                word.lower(),
                words[i][1],#POS
                words[i][2],#Chunk
                '_lower:'+ str(word.islower()),
                '_upper:'+ str(word.isupper()),
                '_digit:'+ str(word.isdigit()),
                '_title:'+ str(word.istitle()),
                '_x:' + str('x' in word),
                #'_y:' + str('y' in word),
                '_long:' + str(len(word)>6),
                'loc:' + str(any(geo.cities) or any(geo.country_mentions)),
                word[-2:],
                word[-3:],
                word_shape(word),
                '-1:'+words[i-1][1] if i>0 else '-1:-',#POS
                '-1:'+words[i-1][2] if i>0 else '-1:-',#Chunk
                '-1:'+words[i-1][0] if i>0 else '-1:-', #previous word
                '+1:'+words[i+1][1] if i<maxi-1 else '+1:-',#POS
                '+1:'+words[i+1][2] if i<maxi-1 else '+1:-',#Chunk
                '+1:'+words[i+1][0] if i<maxi-1 else '+1:-',#next word
                words[i][3] #the label, is only included in the Y, not the X
                ])
    
    return words_upgraded

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

def words2tuples(words):
    symbols = set()
    tag_set = set()
    
    tuples = []
    sentence = []
    for word in words:
        sentence.append((word[0],word[-1]))
        symbols.add(word[0])
        tag_set.add(word[-1])
        if word[1]=='.':
            tuples.append(sentence)
            sentence = []
            
    random.shuffle(tuples)
    return tuples,list(symbols),list(tag_set)