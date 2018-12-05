# -*- coding: utf-8 -*-
"""

@author: Robert Kajnak
"""

import numpy as np
from numbers import Number
from geotext import GeoText
import random
from functools import reduce
from nltk import pos_tag

#%%Read the entries from the file and add them to a list
#with optional filtering
def read_words(filename = 'reuters-train.en'):
    '''returns list of [words per lines]. Blank lines removed'''
    f = open(filename,'r')
    
    words = []
    for line in f:
        l = line.replace('\n','').split(' ')
        if len(l)==1:
            continue
        words.append(l)
    
    f.close()
    return words

#%% Generate new features based on words:

def word_shape(word):
    '''
    Generates a shape string based on the input string.
    Rules: 
        1. capital letters -> X
           smallcase -> x
           digits -> d
           punctuation/other -> unchanged
        2. first and last 2 letters are kept
           for each intermediary character, a sinlge character of the class is kept
    Example:
        a2cdEFG:HI-j -> xdxxXXX:XX-x -> xdxX:-x
        
    Credit for the idea: 
        Information Extraction and Named Entity Recognition,  Michael Collings
        short explanation: https://youtu.be/wxyZTSc2tM0?t=708
    '''
    #convert capital letters into X, non-capitals into x, digits into d and punctuation unchanged
    #.replace() would be slower 
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
            
class feature_list:
    word_itself = True
    POS = True
    word_lowercase = True
    is_lowercase = True
    is_uppercase = True
    is_digit = True
    is_capitalized = True #the first letter
    contains_x = True
    is_long = True  #longer than 6 chars
    is_location = True
    last_2_chars = True
    last_3_chars = True
    word_shape = True
    prev_word = True
    prev_POS = True
    next_word = True
    next_POS = True

def append_features(words, features_to_add = None, is_training_set = True, 
                    is_POS_present_in_words=True):
    '''
    Appends features to the words.
    Keyword arguments:
        words: format list of [word POS class/label]. 
            To omit POS tag set is_POS_present_in_words to False
            to omit class/label set is_training_set to False
        features_to_add:
            None => all possible features added
            otherwise expects feature_list object
        is_training_set: set to False if no output class/labels are provided
        is_training_set: set to False if no POS labels are in the text
    '''

    if features_to_add == None:
        feat = feature_list()
    else:
        feat = features_to_add
    if feat.POS and not is_POS_present_in_words:
        poss = pos_tag([word[0] for word in words])
        
    words_upgraded = []
    maxi = len(words)
    for i in range(maxi):
        word = words[i][0]
        geo = GeoText(word)        
        
        #word+features   
        wpf = []
        if feat.word_itself:    wpf.append(word) #word itself
        if feat.POS:            
                                if is_POS_present_in_words: #POS
                                    wpf.append(words[i][1])
                                else:
                                    wpf.append(poss[i][1])
        if feat.word_lowercase: wpf.append(word.lower()) #word converted to lowercase
        if feat.is_lowercase:   wpf.append('_lower:'+ str(word.islower())) #all letters lowercase
        if feat.is_uppercase:   wpf.append('_upper:'+ str(word.isupper())) #all letters uppercase
        if feat.is_digit:       wpf.append('_digit:'+ str(word.isdigit())) #contains only digits
        if feat.is_capitalized: wpf.append('_title:'+ str(word.istitle())) #first letter capitalized
        if feat.contains_x:     wpf.append('_x:' + str('x' in word or 'X' in word)) #contains 'x' or 'X'
        if feat.is_long:        wpf.append('_long:' + str(len(word)>6)) #longer than 6 characters
        if feat.is_location:    wpf.append('loc:' + str(any(geo.cities) or any(geo.country_mentions))) #city or country
        if feat.last_2_chars:   wpf.append(word[-2:]) #last 2 characters
        if feat.last_3_chars:   wpf.append(word[-3:]) #last 3 characters
        if feat.word_shape:     wpf.append(word_shape(word)) #see word_shape(word) function
        
        if feat.prev_word:      wpf.append('-1:'+words[i-1][0]) if i>0 else wpf.append('-1:-'), #previous word
        if feat.prev_POS:       wpf.append('-1:'+words[i-1][1]) if i>0 else wpf.append('-1:-'),#previous POS
        if feat.next_word:      wpf.append('+1:'+words[i+1][0]) if i<maxi-1 else wpf.append('+1:-'),#next word
        if feat.next_POS:       wpf.append('+1:'+words[i+1][1]) if i<maxi-1 else wpf.append('+1:-'),#next POS
        
        #the label (will be split from X into Y in createDataset(words))
        #Also strips the I or B tag
        if is_training_set:
           if words[i][3]=='O': 
               wpf.append(words[i][3]) 
           else :
               wpf.append(words[i][3][2:])
        words_upgraded.append(wpf)
    
    return words_upgraded

#%%Reshape to the desired one;
#doing this while reading it would increase speed, but this is better for modularity

class translator:
    '''
    converts strings into a unique number
    Example: If called (in this order) on each of the of the elements in the array:
        ['a',b','asdac','b','a'] -> [0, 1, 2, 1, 0]
        [0, 'a' , 2] -> ['a', 0, 'asdac']
        Input can be any non-numeric type. Numbers will be looked up and converted
        back to the original object
    '''
    def __init__(self):
        self.words={}
        self.idx = 0
    def translate(self,word):
        '''
        Works both ways. 
        Number will be converted into the stored object, 
        new objects will be assigned new numbers
        '''
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
                
def create_dataset(words, input_only = False):
    '''
    Converts the strings from the dataset into numbers. 
    (can be used as a pre-cursor) to vectorization (in nltk sense).
    Returns the translator used, the list of numbers and the list of labels 
    for the output classes
    '''

    T = translator()
    
    nw = len(words)
    nf = len(words[0])-1
    
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
#This should however not be the case, since the input should not contain output classes
#the same way as the actual NE label
def one_hot(X, transl):
    '''
    Converts the numbers from the dataset into one-hot encoding. 
    To obtain X and transl, run createDataset(words)
    '''
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
    '''Shuffles while keeping the indices together. a[i]->a[j]=>b[i]->b[j]. a and b mutated'''
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    
class data_wrap:
    '''Shorthand for writing x_train, y_train etc. every time'''
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
    '''returns: x_train,y_train,x_test,y_test'''
    lim = (np.int)(len(Y)*ratio)
    return X[:lim],Y[:lim],X[lim:],Y[lim:]

#%%Construct dictionary for CRF
def sentence_end(word,POS_index):
    if POS_index==None:
        return word[0] in '.?!'
    else:
        return word[POS_index] == '.'
    
def words2dictionary(words,feature_names=None,POS_index = None):
    '''
    Trasnforms word list into senctences (list of list -> list of list of dict)
    Used for CRF model.
    
    Keyword arguments:
        words: list of [x1, x2,...,y]
        feature_names: optional. The dict returned will use this list as keys.
            Having different nr of feature_names and x colums is not supported
        POS_index: if specified, this will be used to identify sentence endings
            The POS character associated to the sentence end should be '.'.
            If left None, '.?!' set will be used to identify sentence ending
    '''
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
        if sentence_end(word,POS_index):
            new_tokens.append(token_sentence)
            token_sentence = []
            new_labels.append(label_sentence)
            label_sentence = []
        #new_words.append(new_word)
        
    return new_tokens, new_labels

def words2tuples(words,feature_used = 0,POS_index = None):
    '''
    Trasnforms word list into senctences (list of list -> list of list of tuple)
    Only one feature is used.
    Used for HMM
    
    Keyword arguments:
        words: list of [x1, x2,...,y]
        features_used: index represents the column from the input
        POS_index: if specified, this will be used to identify sentence endings
            The POS character associated to the sentence end should be '.'.
            If left None, '.?!' set will be used to identify sentence ending
    '''

    
    symbols = set()
    tag_set = set()
    
    tuples = []
    sentence = []
    for word in words:
        sentence.append((word[feature_used],word[-1]))
        symbols.add(word[feature_used])
        tag_set.add(word[-1])
        if sentence_end(word,POS_index):
            tuples.append(sentence)
            sentence = []
            
    random.shuffle(tuples)
    return tuples,list(symbols),list(tag_set)