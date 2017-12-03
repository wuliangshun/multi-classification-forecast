#coding:utf-8

from mlp import process
from mlp import train
from mlp import log
import numpy as np
import os

log_dir = './log'
directory = './model'



import warnings
warnings.filterwarnings("ignore")

def single_test(model_name, x_train_valid, y_train_valid, x_test, y_test):
    # train model and predict
    save_list,pred_train,pred_valid = train.TrainModel.train_simple_model(model_name, x_train_valid, y_train_valid, x_test, y_test, 
                                                                          parameters=train.TrainModel.get_parameters(model_name),     
                                                                          is_grid_search=True, is_step_gridsearch=True, is_prob=True)
    # save model
    log.save_pickle(save_list,model_name)
    return pred_valid

def stack_test(models, ensemble_model, k_fold, x_train_valid, y_train_valid, x_test, y_test):
    
    #layer1
    layer1_list, ensem_x_train, ensem_y_train = train.StackModel.layer1(x_train_valid, y_train_valid, models, k_fold, is_step_gridsearch=True)
            
    #layer2
    layer2_list = train.StackModel.layer2(ensemble_model, ensem_x_train, ensem_y_train)
      
    #predict
    pred_test = train.StackModel.predict(layer1_list, layer2_list, x_test)
       
    # save model
    log.save_pickle([layer1_list,layer2_list],'stacking')
    
    return pred_test
      
def tpot_test(x_train_valid, y_train_valid, x_test, y_test, is_prob):
    # select model
    train.TrainModel.tpot_select_model(x_train_valid,y_train_valid,x_test,y_test)
    print('tpot select model done!')
    
    # train model and predict
    exported_pipeline,pred_test = train.TrainModel.tpot_train_model(x_train_valid,x_test,y_train_valid,y_test, is_prob)
    print('tpot train and predict model done!')
    
    # save model
    log.save_pickle(exported_pipeline,'tpot')
   
    return pred_test

  
def prepare_all_data(path):    
    
    df = process.ProcessData.get_data(path,isdf=True)
    
    X = process.ProcessData.normalization_data(df.ix[:,:-1]).as_matrix()
    Y = df.ix[:,-1].as_matrix()
    
    columns = df.columns.tolist()[:-1]
    
    return X,Y,columns

def split_and_normalization_data(X,Y): 
    
    test_inxs = np.random.randint(0,X.shape[0],int(X.shape[0]/10))
    x_test = X[test_inxs,:]
    y_test = Y[test_inxs]

    x_train_valid = np.delete(X, test_inxs, 0) 
    y_train_valid = np.delete(Y, test_inxs, 0) 
    
    return x_train_valid, y_train_valid, x_test, y_test
    
def predict_data_normalization(X, x_raw_test):
    
    min_np,max_np = process.ProcessData.get_col_min_max(X)
    
    for i in range(x_raw_test.shape[0]) :
        x_raw_test[i,:] = (x_raw_test[i,:] - min_np)/(max_np - min_np)
    
    return x_raw_test
    
    
def train_and_save_models(x_train_valid, y_train_valid, x_test, y_test, models):
    
    '''
    return:
        ytest_num_dict:     test set num for each label
        ytest_accuracy_dict: test set accuracy for each model and each label
    '''
    
    
    ytest_num_dict = {}
    ytest_accuracy_dict = {}
    
    log.log_and_print('-----------------------------------------',log_dir)
    log.log_and_print('Training models...',log_dir)
    
    #ytest_num_dict
    for val in list(set(list(y_test))):
        status = np.where(y_test==val)[0]     
        ytest_num_dict[str(val)] =  status.shape[0]
    
    #tpot
    if 'tpot' in models:
        is_prob = False
        pred_test = tpot_test(x_train_valid, y_train_valid, x_test, y_test, is_prob)
        if is_prob:
            accuracy = train.TrainModel.compute_prob_accuracy(pred_test, y_test,'tpot all :')
        else:
            accuracy = train.TrainModel.compute_accuracy(pred_test, y_test,'tpot all :')
            
        ytest_accuracy_dict['tpot_all'] = accuracy
        
        for val in list(set(list(y_test))):
            status = np.where(y_test==val)[0]     
            log.log_and_print('{0} num : {1}'.format(str(val), str(status.shape[0])),log_dir)
            y_raw_test = y_test[status]
            if is_prob:
                pred_raw_test = pred_test[status,:]
                accuracy = train.TrainModel.compute_prob_accuracy(pred_raw_test, y_raw_test,'tpot for '+str(val)+':')
            else:
                pred_raw_test = pred_test[status]
                accuracy = train.TrainModel.compute_accuracy(pred_raw_test, y_raw_test,'tpot for '+str(val)+':')
            ytest_accuracy_dict['tpot_'+str(val)] = accuracy
        
    #xgb
    if 'xgb' in models:
        pred_test = single_test('xgb',x_train_valid, y_train_valid, x_test, y_test)
        accuracy = train.TrainModel.compute_prob_accuracy(pred_test, y_test, 'xgb all :')
        
        ytest_accuracy_dict['xgb_all'] = accuracy
        
        for val in list(set(list(y_test))):
            status = np.where(y_test==val)[0]
            log.log_and_print('{0} num : {1}'.format(str(val), str(status.shape[0])),log_dir)
            pred_raw_test = pred_test[status,:]
            y_raw_test = y_test[status]
            accuracy = train.TrainModel.compute_prob_accuracy(pred_raw_test, y_raw_test, 'xgb for '+str(val)+':')
            ytest_accuracy_dict['xgb_'+str(val)] = accuracy
    
    #stacking   
    if 'stacking' in models:
        pred_test = stack_test(models = ['xgb','lgb_sk'],    # 'knn','svm','cb','lgb',
                               ensemble_model = 'lr',
                               k_fold = 3,
                               x_train_valid=x_train_valid,
                               y_train_valid=y_train_valid,
                               x_test=x_test,
                               y_test=y_test)
        accuracy = train.TrainModel.compute_prob_accuracy(pred_test, y_test, 'stacking all:')
        
        ytest_accuracy_dict['stacking_all'] = accuracy
    
        for val in list(set(list(y_test))):
            status = np.where(y_test==val)[0]
            log.log_and_print('{0} num : {1}'.format(str(val), str(status.shape[0])),log_dir)
            pred_raw_test = pred_test[status,:]
            y_raw_test = y_test[status]
            accuracy = train.TrainModel.compute_prob_accuracy(pred_raw_test, y_raw_test, 'stacking for '+str(val)+':')
            ytest_accuracy_dict['stacking_'+str(val)] = accuracy
        
    return ytest_num_dict,ytest_accuracy_dict
    

def get_models_feature_importance(columns):
    
    '''
    return:
        importance_result_dict: 
            key: feature name
            value: feature weight
    '''
    
    log.log_and_print('-----------------------------------------',log_dir)
    log.log_and_print('Get models feature importance...',log_dir)
    
    importance_list = []
    
    
    if os.path.exists(os.path.join(directory,'xgb.pickle')):
        obj1 = log.load_pickle('xgb',directory)
        importance_list.append( obj1[0].get_importance( obj1[1]))
    
    if os.path.exists(os.path.join(directory,'stacking.pickle')):
        obj2 = log.load_pickle('stacking',directory)
        [layer1_list,layer2_list] = obj2
        
        for i in range(len(layer1_list)):
            for j in range(len(layer1_list[0])):    
                importance_list.append(layer1_list[i][j][0].get_importance( layer1_list[i][j][1]))
    
    #compute   
          
    for i in range(len(importance_list)):
        importance_list[i] = importance_list[i]/sum(importance_list[i])
            
    importance_matrix = np.array(importance_list)
    
    result = importance_matrix.mean(axis=0)
    
    indices = np.argsort(result)[::-1]

    feature_names = columns 

    importance_result_dict = {}
    for f in range(len(result)):
        log.log_and_print("%d | %s | %.2f" % (f + 1, feature_names[indices[f]], result[indices[f]]),log_dir)
        importance_result_dict[feature_names[indices[f]]] = result[indices[f]]
        
    return importance_result_dict
            
    

def new_predict(x_test):
    
    '''
    return:
        pred_dict: predict result for each model
    '''
    
    log.log_and_print('-----------------------------------------',log_dir)
    log.log_and_print('New predict...',log_dir)
    
    pred_dict = {}
    
    if os.path.exists(os.path.join(directory,'xgb.pickle')):
        obj1 = log.load_pickle('xgb',directory)
        pred1 = obj1[0].predict(obj1[1],x_test)
        log.log_and_print(u'xgb model predict:%.2f |\t%.2f |\t%.2f'%(pred1[0,0],pred1[0,1],pred1[0,2]),log_dir)
        pred_dict['xgb'] = pred1
    
    if os.path.exists(os.path.join(directory,'stacking.pickle')):
        obj2 = log.load_pickle('stacking',directory)
        pred2 = train.StackModel.predict(obj2[0], obj2[1], x_test)
        log.log_and_print(u'stack model predict:%.2f |\t%.2f |\t%.2f'%(pred2[0,0],pred2[0,1],pred2[0,2]),log_dir)
        pred_dict['stacking'] = pred2
    
    if os.path.exists(os.path.join(directory,'tpot.pickle')):
        obj3 = log.load_pickle('tpot',directory)
        try:
            pred3 = obj3.predict_proba(x_test)
            log.log_and_print(u'tpot  model predict:%.2f |\t%.2f |\t%.2f'%(pred3[0,0],pred3[0,1],pred3[0,2]),log_dir)
        except:
            pred3 = obj3.predict(x_test)
            log.log_and_print(u'tpot  model predict:%d'%(pred3),log_dir)
        pred_dict['tpot'] = pred3
        
   
    return pred_dict
    


def new_predict_matrix(x_test,y_test):
    
    '''
    return:
        pred_dict: predict result for each model
    '''
    
    log.log_and_print('-----------------------------------------',log_dir)
    log.log_and_print('Test dataset ...',log_dir)
    
    pred_dict = {}
    
    #tpot
    if os.path.exists(os.path.join(directory,'tpot.pickle')):        
        obj3 = log.load_pickle('tpot',directory)
        try:
            pred3 = obj3.predict_proba(x_test)           
            accuracy_tpot = train.TrainModel.compute_prob_accuracy(pred3, y_test, 'tpot')        
        except:
            pred3 = obj3.predict(x_test)           
            accuracy_tpot = train.TrainModel.compute_accuracy(pred3, y_test, 'tpot')
        pred_dict['tpot'] = accuracy_tpot
    
    #xgb
    if os.path.exists(os.path.join(directory,'xgb.pickle')):
        obj1 = log.load_pickle('xgb',directory)
        pred1 = obj1[0].predict(obj1[1],x_test)    
        accuracy_xgb = train.TrainModel.compute_prob_accuracy(pred1, y_test, 'xgb')
        pred_dict['xgb'] = accuracy_xgb
    
    #stacking
    if os.path.exists(os.path.join(directory,'stacking.pickle')):
        obj2 = log.load_pickle('stacking',directory)
        pred2 = train.StackModel.predict(obj2[0], obj2[1], x_test)
        accuracy_stacking = train.TrainModel.compute_prob_accuracy(pred2, y_test, 'stacking')
        pred_dict['stacking'] = accuracy_stacking
    
    return pred_dict
 
def select_columns(cols,X,Y,columns,method):    
    '''
    method:
        'by_inx':  by column index
        'by_name': by column names
    '''
    if method == 'by_inx':
        col_inxs = cols
    elif method == 'by_name':        
        col_inxs = []
        for col in cols:
            col_inxs.append(columns.index(col))
    else:
        raise ValueError('wrong input method !')        
    X = X[:,col_inxs]
    columns = [columns[inx] for inx in col_inxs]
    return X,Y,columns   
 


    
    
    
    
    
    
    