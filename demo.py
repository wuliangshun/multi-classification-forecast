# -*- coding: utf-8 -*-
"""
demo
"""
import time
import numpy as np
import mlp 




if __name__=='__main__':
    
    import sys  
    
    if len(sys.argv) == 5:
        mlp.main.log_dir = sys.argv[1]
        mlp.main.directory = sys.argv[2]
        data_path = sys.argv[3]
        models = sys.argv[4]       
    else:    
        mlp.main.log_dir = './log'
        mlp.main.directory = './model'
        data_path = './data/data_process.csv'
        models = ['tpot','xgb','stacking']
    
    starttime = time.time()
    
    # prepare data
    X,Y,columns =  mlp.main.prepare_all_data(path=data_path)    
    
    # select columns    
    #cols = [name1,name2,...]   or [0,1,2,3,4,6,7,9,11,12] 
    #X,Y,columns = select_columns(cols,X,Y,columns,method='by_name')
    
    # split data
    x_train_valid, y_train_valid, x_test, y_test = mlp.main.split_and_normalization_data(X,Y)
  
    # train and save models
    ytest_num_dict,ytest_accuracy_dict = mlp.main.train_and_save_models(x_train_valid, y_train_valid, x_test, y_test, models)
    
    # output average feature importance
    importance_result_dict = mlp.main.get_models_feature_importance(columns)

    # predict one sample
    x_raw = mlp.process.ProcessData.get_data(csv_path=data_path,isdf=False)[:,:-1]    
    x_raw_test = np.array([x_raw[0,:]])    
    x_nor_test = mlp.main.predict_data_normalization(x_raw, x_raw_test)    
    pred_dict = mlp.main.new_predict(x_nor_test)
    
    # predict multiple samples
    x_raw = mlp.process.ProcessData.get_data(csv_path=data_path,isdf=False)[:,:-1]    
    for val in list(set(list(Y))):
        status = np.where(Y==val)[0]
        x_raw_test = x_raw[status,:] 
        y_raw_test = Y[status]
        # print sample num of every value
        mlp.log.log_and_print('{0} num : {1}'.format(str(val), str(status.shape[0])))      
        # normalization
        x_nor_test = mlp.main.predict_data_normalization(x_raw, x_raw_test)  
        # predict and print accuracy
        pred_dict = mlp.main.new_predict_matrix(x_nor_test,y_raw_test)
    
    #print used time
    mlp.log.log_and_print('Total time:{} seconds'.format(time.time() - starttime))
