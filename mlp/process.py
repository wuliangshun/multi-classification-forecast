# -*- coding: utf-8 -*-
"""
Process Data
"""
from sklearn import cross_validation
import pandas as pd
import numpy as np



class ProcessData:
    
    def __init__(self):
        pass
    
    @staticmethod
    def get_data(csv_path,isdf=False):
        
        try:
            df = pd.read_csv(csv_path)
        except:
            df = pd.read_csv(csv_path,encoding='gbk')
            
        df = df.dropna(axis=0)
        data = df.as_matrix()
        if isdf:
            return df
        else:
            return data
        
    @staticmethod
    def get_col_min_max(matrix):       
        min_np = np.array([np.min(matrix[:,j]) for j in range(matrix.shape[1])])
        max_np = np.array([np.max(matrix[:,j]) for j in range(matrix.shape[1])])
        return min_np,max_np
  
    @staticmethod
    def normalization_data(df):
        return df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    
    
    @staticmethod
    def one_hot_encoder(y):
        s = list(set(list(y)))
        vec = [0.0] * len(s)
        result = []
        for j in y:
            inx = s.index(j)
            temp_vec = vec.copy()
            temp_vec[inx] = 1
            result.append(temp_vec)
        return np.array(result)
    
    @staticmethod
    def split_train_valid( X, Y, options):
        
        if options is not None:
            for para in options:
                if para == 'valid_size':
                    valid_size = options[para]
                if para == 'train_size':
                    valid_size = options[para]
                if para == 'random_state':
                    valid_size = options[para]
        else:
            valid_size = 0.33
            random_state = 43
            
        x_train, x_valid, y_train, y_valid = cross_validation.train_valid_split(X, Y, valid_size=valid_size, random_state=random_state)
        
        return x_train, x_valid, y_train, y_valid
    
    @staticmethod
    def cross_validation_k_fold( X, Y, is_final=True, n_folds=4, shuffle=False, random_state=None):
         
        n=len(Y)
        kf = cross_validation.KFold(n=n,n_folds=n_folds,shuffle=shuffle,random_state=random_state)
        
        x_train_list = []
        x_valid_list = []
        y_train_list = []
        y_valid_list = []
        for train_index,valid_index in kf:            
            x_train,x_valid = X[train_index],X[valid_index]
            y_train,y_valid = Y[train_index],Y[valid_index]
            x_train_list.append(x_train)
            x_valid_list.append(x_valid)
            y_train_list.append(y_train)
            y_valid_list.append(y_valid)
            
        if is_final:
            return x_train, x_valid, y_train, y_valid
        else:
            return x_train_list,x_valid_list,y_train_list,y_valid_list
        
    
 