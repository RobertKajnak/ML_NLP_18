# -*- coding: utf-8 -*-
"""

@author: Robert Kajnak
"""
from preprocessing import (readWords, appendFeatures,
                           createDataset, oneHot, data_wrap, words2dictionary, 
                           shuffle_parallel, split_tr)
from models import (NB, LR, SVM, HMM, CRF)


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
    