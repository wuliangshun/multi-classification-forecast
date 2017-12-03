# -*- coding: utf-8 -*-
"""
Log
"""

import time
import os
    
def log(content,log_dir='./log'):
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir) 
    filename=os.path.join(log_dir,'log.txt')
    with open(filename,'a') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) + ':\t' + content)
        f.write('\n')
        
def log_and_print(content,log_dir='./log',encoding='gbk'):
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir) 
    filename=os.path.join(log_dir,'log.txt')
    with open(filename,'a') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) + ':\t' + content)
        f.write('\n')
    print(content)
    
def save_pickle(obj,name,directory='./model'):
    if not os.path.isdir(directory):
        os.makedirs(directory) 
    import pickle  
    with open(os.path.join(directory,name+'.pickle'),'wb') as f:
        pickle.dump(obj, f)
        
def load_pickle(name,directory='./model'):
    if not os.path.isdir(directory):
        os.makedirs(directory) 
    import pickle  
    with open(os.path.join(directory,name+'.pickle'),'rb') as f:
        return pickle.load(f)
    
