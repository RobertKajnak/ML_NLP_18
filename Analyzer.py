# -*- coding: utf-8 -*-
"""

@author: Robert Kajnak
"""
from preprocessing import (read_words, append_features,
                           create_dataset, one_hot, data_wrap, words2dictionary, 
                           shuffle_parallel, split_tr,words2tuples)
from models import (NB, LR, SVM,HMM_old, HMM, CRF)


#%%Handy method for testing the models
def run_models(words, models, verbose, train=True, test=True):
    '''
    Runs all the models that are specified with the specified word set.
    It runs all preporocessing steps necessary for the models specified
    Note: If a model is specified twice, it will be run twice, but the preprocessing
    on the input data will not(useful to test for model parameter initialization)
    
    Returns a list containing the the objects of the models used, 
        the outputs they predicted and 
        the sklearn classification reports (dictionary format), 
        in the order where they were provided
        
    Keyword arguments:
        words: list of list of words and features. 
            Format: n*m. n=nr of words, m=nr features + expected output (single)
        models: a string containing the model names. Order is not important.
            Possible models are: NB, LR, SVM, HMM, CRF. Coming soon: CNN
            If a model is specified twice, it will be run twice. The input is
            randomized only once, where applicable
        veboose: 0: print nothing
                1: print results
                2: print status messages:
                3: print both
    '''
    # Preparing data for one-hot encodign -- converts strings into integers
    if any(i in models for i in ['HMM_old','NB','LR','SVM']):
        verbose|2 and print('Initial pre-processing...')
        X,Y,transl,labels_num,labels_name = create_dataset(words)
        
    # Algoirthms using non-randomized, one-hot data:HMM
    if 'HMM_old' in models:
        verbose|2 and print('Preprocessing data for HMM, old version...')
        X_onehot_ord = one_hot(X,transl)
        x_train_oh,y_train_oh,x_test_oh,y_test_oh = split_tr(X_onehot_ord,Y,0.8)
        data_ordered_oh = data_wrap(x_train_oh,y_train_oh,x_test_oh,y_test_oh,transl,labels_num,labels_name)
    
    #Algorithm uses sentences (list of list of tuples): HMM
    if 'HMM' in models:
        verbose|2 and print('Preprocessing data for HMM...')
        sentences_hmm, symbols, tag_set = words2tuples(words)
        _,y_train,_,y_test = split_tr([],sentences_hmm,0.8)
        x_test = [[tup[0] for tup in sentence] for sentence in y_test]
        y_test = [[tup[1] for tup in sentence] for sentence in y_test]
        data_hmm = data_wrap(None,y_train,x_test,y_test)
    
    # Algorithms using shuffled, one-hot data:NB,LR,SVM
    if any(i in models for i in ['NB','LR','SVM']):
        verbose|2 and print('Preprocessing data for NB, LR and/or SVM...')
        shuffle_parallel(X,Y)
        X_onehot_sh = one_hot(X,transl)
        x_train_oh_sh,y_train_oh_sh,x_test_oh_sh,y_test_oh_sh = split_tr(X_onehot_sh,Y,0.8)
        data_shuffled = data_wrap(x_train_oh_sh,y_train_oh_sh,x_test_oh_sh,y_test_oh_sh,transl,labels_num,labels_name)

    #Ordered, using sentences (list of list of dict): CRF
    if 'CRF' in models:
        verbose|2 and print('Preprocessing data for CRF...')
        tokens_dict,labels_dict = words2dictionary(words)#,['token','POS','segment'])
        
        tokens_train,labels_train,tokens_test,labels_test = split_tr(tokens_dict,labels_dict,0.8)
        data_dictionary = data_wrap(tokens_train,labels_train,tokens_test,labels_test)
    
    model_objects = []
    model_results = []
    model_predictions = []
    #removes clutter when calling the functions separately
    #Using a list of function handlers could also be used, but I find that to be 
    #less intuitive
    def _add_to_output(model_y_pred):
        model_objects.append(model_y_pred[0])
        model_results.append(model_y_pred[1])
        if (len(model_y_pred)>2):
            model_predictions.append(model_y_pred[2])
        
    #Run each of the models from the paramters, while KEEPING THE ORDER they were called in
    #and append it to the return lists
    for model in models:
        if 'HMM_old' in model:
            verbose|2 and print('Running HMM from hmmlearn package...')
            _add_to_output(HMM_old(data_ordered_oh,verbose|1))
            
        if 'HMM' in model:
            verbose|2 and print('Running HMM from nltk...')
            _add_to_output(HMM(data_hmm,symbols,tag_set,verbose|1))
            
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
            
    return model_objects,model_results,model_predictions
            
#%% Main
if __name__ == "__main__":
    print('Loading document...')
    words = read_words('reuters-train.en')
    print('Adding features...')
    words = append_features(words, is_POS_present_in_words=True, is_training_set=True)
    
    #comment any of these to not run it. The necessary pre-processing steps for
    #that model will also be skipped
    models=[
            'NB',
            'LR',
            'SVM', 
            'HMM',
            'CRF'
            ]
    models,predictions,reports = run_models(words, models, verbose = 3, 
                                            train=True, test=True)
    averages = [report['weighted avg']['f1-score'] for report in reports]
    
    #History-like feature. Appends the f1-weighted-average to the variable
    #if the variable doesn't exist, an empty list is instantiated
    try:
        avg_settings
    except:
        avg_settings = []
    avg_settings.append(averages)
    if (len(avg_settings)>1) and len(avg_settings[-1])==len(avg_settings[-2]):
        print('Change since last run (this-last):')
        #this is not a keyword here. <giggles to himself>
        for this, last in zip(avg_settings[-1],avg_settings[-2]):
            print('{0:.3f}'.format(this-last))
            