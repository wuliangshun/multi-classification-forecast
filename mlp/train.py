# -*- coding: utf-8 -*-
"""
Train
"""
from sklearn.grid_search import GridSearchCV 
import time 
import numpy as np
from mlp import log
from mlp import models
from mlp import process


class TrainModel:
    
    def __init__(self):
        pass
    
    @staticmethod
    def get_parameters(model_name, is_get_group=False, train_seed=0):
        
        parameters = {}
        
        if not is_get_group:
            
            if model_name == 'knn':
                parameters = {
                        'n_neighbors':2
                        }
            elif model_name == 'svm':
                parameters = {
                        'probability':True,
                        'kernel':'rbf', 
                        'gamma':1e-3,
                        'C':10}        
            elif model_name == 'lr':
                parameters = {'C': 1.0,
                      'class_weight': None,
                      'dual': False,
                      'fit_intercept': True,
                      'intercept_scaling': 1,
                      'max_iter': 100,
                      'multi_class': 'multinomial',
                      #'multi_class': 'ovr',
                      'n_jobs': 2,
                      'penalty': 'l2',
                      'solver': 'sag',
                      'tol': 0.0001,
                      'random_state': train_seed,
                      'verbose': 1,
                      'warm_start': False}
            elif model_name == 'rf':
                parameters = {
                      'bootstrap': False,
                      'class_weight': None,
                      'criterion': 'gini',
                      'max_depth': 2,
                      'max_features': 0.8,
                      'max_leaf_nodes': None,
                      'min_impurity_decrease': 0.0,
                      'min_samples_leaf': 6,
                      'min_samples_split': 18,
                      'min_weight_fraction_leaf': 0.0,
                      'n_estimators': 100,
                      'n_jobs': -1,
                      'oob_score': True,
                      'random_state': train_seed,
                      'verbose': 2,
                      'warm_start': False}
            elif model_name == 'et':
                parameters = {
                      'bootstrap': False,
                      'class_weight': None,
                      'criterion': 'gini',                      
                      'max_features': 0.8,                    
                      'min_impurity_decrease': 0.0,
                      'min_samples_leaf': 6,
                      'min_samples_split': 18,
                      'min_weight_fraction_leaf': 0.0,
                      'n_estimators': 100,
                      'n_jobs': -1,
                      'oob_score': True,
                      'random_state': train_seed,
                      'verbose': 2,
                      'warm_start': False}
            elif model_name == 'ab':
                parameters = {'bootstrap': True,
                         'class_weight': None,
                         'criterion': 'gini',
                         'max_depth': 2,
                         'max_features': 7,
                         'max_leaf_nodes': None,
                         'min_impurity_decrease': 0.0,
                         'min_samples_leaf': 6,
                         'min_samples_split': 18,
                         'min_weight_fraction_leaf': 0.0,
                         'n_estimators': 100,
                         'n_jobs': -1,
                         'oob_score': True,
                         'random_state': train_seed,
                         'verbose': 2,
                         'warm_start': False}
            elif model_name == 'gb':
                parameters = {'criterion': 'friedman_mse',
                      'init': None,
                      'learning_rate': 0.05,
                      'loss': 'deviance',
                      'max_depth': 25,
                      'max_features': 'auto',
                      'max_leaf_nodes': None,
                      'min_impurity_decrease': 0.0,
                      'min_impurity_split': None,
                      'min_samples_leaf': 50,
                      'min_samples_split': 1000,
                      'min_weight_fraction_leaf': 0.0,
                      'n_estimators': 200,
                      'presort': 'auto',
                      'random_state': train_seed,
                      'subsample': 0.8,
                      'verbose': 2,
                      'warm_start': False}
            elif model_name == 'xgb':
                parameters = {  
                               'silent':0,            # 是否打印输出                               
                               'learning_rate': 0.1,
                               'n_estimators':100,    # 树的个数
                               'max_depth':5,         # 构建树的深度，越大越容易过拟合
                               'min_child_weight':1,  # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
                               'gamma':0,             # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子
                               'subsample': 0.8,      # 随机采样训练样本 训练实例的子采样比
                               'max_delta_step':0,    # 最大增量步长，我们允许每个树的权重估计
                               'colsample_bytree':0.8, # 生成树时进行的列采样 
                               'reg_lambda':1,        # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合
                               'reg_alpha':0,         # L1 正则项参数
                               'objective': 'multi:softmax', # 二分类：'binary:logistic','multi:softprob'                               
                               'nthread':4,
                               'scale_pos_weight':1,  # 如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。平衡正负权重
                               'seed':train_seed}
            elif model_name == 'lgb_sk':
                parameters = {'learning_rate': 0.003,
                      'boosting_type': 'gbdt',        # traditional Gradient Boosting Decision Tree.
                      # 'boosting_type': 'dart',        # Dropouts meet Multiple Additive Regression Trees.
                      # 'boosting_type': 'goss',        # Gradient-based One-Side Sampling.
                      # 'boosting_type': 'rf',          # Random Forest.
                      'num_leaves': 80,               # <2^(max_depth)
                      'max_depth': 7,                 # default=-1
                      'n_estimators': 50,
                      'max_bin': 1005,
                      'subsample_for_bin': 1981,
                      'objective': 'softmax',
                      'min_split_gain': 0.,
                      'min_child_weight': 1,
                      'min_child_samples': 0,
                      'subsample': 0.8,
                      'subsample_freq': 5,
                      'colsample_bytree': 0.8,
                      'reg_alpha': 0.5,
                      'reg_lambda': 0.5,
                      'silent': False,
                      'random_state': train_seed}
            elif model_name == 'lgb':
                parameters = {                      
                      'application': 'softmax',
                      'objective': 'softmax',
                      'boosting': 'gbdt',                   # gdbt,rf,dart,goss
                      'learning_rate': 0.003,               # default=0.1
                      'num_leaves': 10,                     # default=31       <2^(max_depth)
                      'max_depth': 5,                       # default=-1
                      'min_data_in_leaf': 20,               # default=20       reduce over-fit
                      'min_sum_hessian_in_leaf': 1e-3,      # default=1e-3     reduce over-fit
                      'feature_fraction': 0.5,              # default=1
                      'feature_fraction_seed': train_seed,  # default=2
                      'bagging_fraction': 0.8,              # default=1
                      'bagging_freq': 2,                    # default=0        perform bagging every k iteration
                      'bagging_seed': train_seed,           # default=3
                      'lambda_l1': 0,                       # default=0
                      'lambda_l2': 0,                       # default=0
                      'min_gain_to_split': 0,               # default=0
                      'max_bin': 225,                       # default=255
                      'min_data_in_bin': 1,                 # default=5
                      'metric': 'multi_logloss',            # 'binary_logloss'
                      'num_threads': -1,
                      'verbosity': 1,
                      'early_stopping_rounds': 50,          # default=0
                      'seed': train_seed}
            elif model_name == 'cb':
                parameters = {'iterations': 50,
                      'learning_rate': 0.003,
                      'depth': 8,                            # Depth of the tree.
                      'l2_leaf_reg': 0.3,                      # L2 regularization coefficient.
                      'rsm': 1,                              # The percentage of features to use at each iteration.
                      'bagging_temperature': 0.9,              # Controls intensity of Bayesian bagging. The higher the temperature the more aggressive bagging is.
                      'loss_function':'MultiClass',            #    'Logloss',
                      'feature_border_type': 'MinEntropy',
                      'fold_permutation_block_size': 1,
                      'od_pval': None,                       # Use overfitting detector to stop training when reaching a specified threshold.
                      'od_wait': None,                       # Number of iterations which overfitting detector will wait after new best error.
                      'od_type': 'IncToDec',                 # Type of overfitting detector which will be used in program.
                      'gradient_iterations': None,           # The number of gradient steps when calculating the values in leaves.
                      'leaf_estimation_method': 'Gradient',  # The method used to calculate the values in leaves.
                      'thread_count': None,                  # Number of parallel threads used to run CatBoost.
                      'random_seed': train_seed,
                      'use_best_model': False,               # To limit the number of trees in predict() using information about the optimal value of the error function.
                      'verbose': True,
                      'ctr_description': None,               # Binarization settings for categorical features.
                      'ctr_border_count': 16,                # The number of partitions for Categ features.
                      'ctr_leaf_count_limit': None,          # The maximum number of leafs with categorical features.
                      'priors': None,                        # Use priors when training.
                      'has_time': False,                     # To use the order in which objects are represented in the input data.
                      'name': 'experiment',
                      'ignored_features': None,
                      'train_dir': None,
                      'custom_loss': None,
                      'classes_count':4,
                      'class_weights': None}
            elif model_name == 'dnn':
                parameters = { 
                      'epochs':2000,
                      'unit_number':20,
                      'n_classes':4,
                      'learning_rate':0.01,
                      'learning_rate_decay':0.99,
                      'regularization_rate':0.0001, 
                      'batch_size':60,
                      'display_step':10
                    }
        else:
            
            if model_name == 'rnn':
                parameters = {
                        'n_neighbors':[2,4,6]
                        }
            elif model_name == 'svm':
                parameters = {
                        'probability':[True],
                        'kernel':['rbf'], 
                        'gamma':[1e-3, 1e-4],
                        'C':[1, 10, 100, 1000]}  
            elif model_name == 'lr':
                parameters = {
                      'C': [0.2,0.5,0.8,1],                      
                      'intercept_scaling': [0,1],
                      'max_iter': [50,100,200,500],   
                      'tol': [0.0001,0.001,0.005,0.01],
                      }  
            elif model_name == 'ab':
                parameters = {
                        'learning_rate': [0.002, 0.003, 0.005],
                         'n_estimators': [50, 100]
                      }
            elif model_name == 'rf':
                parameters = {
                           # 'n_estimators': [30, 31, 32],
                           'max_depth': [2, 3],
                           # 'max_features': [6, 7],
                           'min_samples_leaf': [286, 287],
                           'min_samples_split': [3972, 3974, 3976, 3978]
                           }
            elif model_name == 'et':
                parameters = {
                           'n_estimators': [30, 40, 50],
                           'max_depth': [5, 6],
                           'max_features': [6, 7],
                           'min_samples_leaf': [200, 250, 300],
                           'min_samples_split': [3000, 3500, 4000]
                           }                
            elif model_name == 'gb':
                parameters = {
                           'n_estimators': [20, 50, 100],
                           'learning_rate': [0.05, 0.2, 0.5],
                           'max_depth': [5, 10, 15],
                           'max_features': [6, 8, 10],
                           'min_samples_leaf': [6,10,50,100,300],
                           'min_samples_split': [100,300, 400, 1000],
                           'subsample': [0.6, 0.8, 1]
                      } 
            elif model_name == 'xgb':
                parameters = {        
                  'colsample_bytree':[0.6,0.8,0.9],
                  'n_estimators':[20,50,100,120,140,180],
                  'max_depth':[1,2,3,5,7], 
                  'min_child_weight':[1,2,3,4,5,6,10,15],
                  'learning_rate':[0.001,0.002,0.005,0.01,0.1],
                  'gamma':[0,0.1,0.2,0.4],
                  'subsample':[0.2,0.3,0.5,0.8,0.9],
                  'max_delta_step':[0,0.1,0.2],                  
                  'reg_lambda':[1,1.2,1.5], 
                   'reg_alpha':[0,0.1,0.2]                   
                  }
            elif model_name == 'lgb_sk':
                 parameters ={
                           'boosting_type':['gbdt'], 
                           'objective':['softmax'],
                           'learning_rate': [0.002, 0.005, 0.01, 0.1],
                           'n_estimators': [10, 30, 60, 90, 100, 120],
                           'num_leaves': [32, 64, 128],             # <2^[max_depth]
                           'colsample_bytree': [0.6, 0.8, 0.1, 1],
                           'max_depth': [5, 6, 8, 10],                    
                      'max_bin':  [255, 355, 455],
                      'subsample_for_bin': [100,500,1000,2000,8000,15000,200000],     
                      'min_child_samples':[2,10,20],
                      'min_split_gain': [0.0,0.1,0.2],
                      'min_child_weight': [0.001,0.01,0.1],
                      'subsample': [0.8,0.5,1,0.7],
                      'subsample_freq': [1, 2, 4, 6, 8],                    
                      'reg_alpha': [0.5,0.6,0.7,0.8],
                      'reg_lambda': [0.3,0.5,0.6,0.7,0.8],
                      }
            elif model_name == 'cb':
                parameters = {
                      'iterations': [50],
                      'learning_rate': [0.003],
                      'depth': [8],                            # Depth of the tree.
                      'l2_leaf_reg': [0.3],                      # L2 regularization coefficient.
                      'rsm': [1],                              # The percentage of features to use at each iteration.
                      'bagging_temperature': [0.9],              # Controls intensity of Bayesian bagging. The higher the temperature the more aggressive bagging is.                     
                      'fold_permutation_block_size': [1]
                      }
        return parameters
            
    @staticmethod
    def get_model_object(model_name, x_train, y_train, x_valid):
            
        if model_name == 'knn':
            model = models.KNearestNeighbor(x_train, y_train, x_valid)
        elif model_name == 'lr':
            model = models.LRegression(x_train, y_train, x_valid)
        elif model_name == 'rf':
            model = models.RandomForest(x_train, y_train, x_valid)
        elif model_name == 'svm':
            model = models.SupportVectorClustering(x_train, y_train, x_valid)
        elif model_name == 'gs':
            model = models.Gaussian(x_train, y_train, x_valid)
        elif model_name == 'dt':
            model = models.ExtraTrees(x_train, y_train, x_valid)
        elif model_name == 'et':
            model = models.DecisionTree(x_train, y_train, x_valid)
        elif model_name == 'ab':
            model = models.AdaBoost(x_train, y_train, x_valid)
        elif model_name == 'gb':
            model = models.GradientBoosting(x_train, y_train, x_valid)
        elif model_name == 'xgb':
            model = models.SKLearnXGBoost(x_train, y_train, x_valid)
        elif model_name == 'lgb_sk':
            model = models.SKLearnLightGBM(x_train, y_train, x_valid)
        elif model_name == 'lgb':
            model = models.LightGBM(x_train, y_train, x_valid, 1000000000000)
        elif model_name == 'cb':
            model = models.CatBoost(x_train, y_train, x_valid)
           
        else:
            raise ValueError('no model match error!')
            return None
        return model
    
    
    @staticmethod
    def train_simple_model(model_name, x_train, y_train, x_valid, y_valid, parameters, is_grid_search=True, is_step_gridsearch=True, is_prob=True):
        
        model_obj = TrainModel.get_model_object(model_name, x_train, y_train, x_valid)
        
        clf = model_obj.get_clf( TrainModel.get_parameters(model_name))
        
        if is_grid_search:
            if is_step_gridsearch:
                # step by step grid search
                group_paras = TrainModel.get_parameters(model_name, is_get_group=True)
                turned_paras = {}
                for key in group_paras:
                    turned_paras[key] = group_paras[key]
                    best_estimator,best_parameters = TrainModel.grid_search_cv(clf, x_train, y_train, turned_paras)
                    turned_paras[key] = [best_parameters[key]]
            else:
                best_estimator,best_parameters = TrainModel.grid_search_cv(clf, x_train, y_train,  TrainModel.get_parameters(model_name, is_get_group=True))
        else:
            best_estimator = model_obj.fit(x_train, y_train, x_valid, y_valid, parameters) 
        
        pred_train = model_obj.predict(best_estimator, x_train, is_prob=is_prob)
        pred_valid = model_obj.predict(best_estimator, x_valid, is_prob=is_prob)
        
        return [model_obj,best_estimator],pred_train,pred_valid
    
    
    @staticmethod
    def train_dnn_model(x_train, y_train, x_valid, y_valid, parameters):
        
        y_train = process.ProcessData.one_hot_encoder(y_train)    
        y_valid = process.ProcessData.one_hot_encoder(y_valid)
        
        model = models.DeepNeuralNetworks(x_train, y_train, x_valid, y_valid, parameters) 
       
        model.train()
        
        return model   
        
    @staticmethod
    def grid_search_cv(clf, x_train, y_train, param_grid, scoring='neg_log_loss', cv=3, verbose=1, n_jobs=2,iid=True,refit=True):
        """
            estimator: e.g. estimator=RandomForestClassifier()
            param_grid:  {'':[],'':[]} 或 [{'':[],'':[]},{'':[],'':[]}]
            scoring: 准确度评价标准，默认None,这时需要使用score函数；或者如
                    Classification:'roc_auc','neg_log_loss','precision', 'recall','accuracy','average_precision','f1','f1_micro','f1_macro','f1_weighted','f1_samples'
                    Clustring:'adjusted_rand_score'
                    Regression:'neg_mean_absolute_error','neg_mean_square_error','neg_median_absolute_error','r2'
            cv: 交叉验证参数，默认None，使用三折交叉验证，指定fold数量，默认为3
            verbose：日志冗长度，int：冗长度，0：不输出训练过程，1：偶尔输出，>1：对每个子模型都输出
            n_jobs: 并行数，int：个数,-1：跟CPU核数一致, 1:默认值。
            iid:默认True,为True时，默认为各个样本fold概率分布一致，误差估计为所有样本之和，而非各个fold的平均。
            refit :默认为True,程序将会以交叉验证训练集得到的最佳参数，重新对所有可用的训练集与开发集进行，作为最终用于性能评估的最佳模型参数。即在搜索参数结束后，用最佳参数结果再次fit一遍全部数据集。
        """
        start_time = time.time()
        grid_search_model = GridSearchCV(estimator=clf,
                                     param_grid=param_grid,
                                     scoring=scoring,
                                     verbose=verbose,
                                     n_jobs=-n_jobs,
                                     cv=cv,
                                     iid=iid,
                                     refit=refit)
        
        #print('Grid Searching...')
        
        grid_search_model.fit(x_train, y_train)
        
        best_estimator =  grid_search_model.best_estimator_
        best_parameters = grid_search_model.best_estimator_.get_params()
        
        #best_score = grid_search_model.best_score_
        
        #print('Best score: %0.6f' % best_score)
        
        print('Best parameters set:')

        for param_name in sorted(param_grid.keys()):
            print('\t%s: %r' % (param_name, best_parameters[param_name]))  
        
        total_time = time.time() - start_time
        
        print('Total time:{}\n'.format(total_time))
        
        return best_estimator,best_parameters
        
    @staticmethod
    def get_k_fold_sample(X, Y, k_fold):
        
         x_train_list,x_valid_list,y_train_list,y_valid_list = process.ProcessData.cross_validation_k_fold( X, Y, is_final=False, n_folds=k_fold, shuffle=False, random_state=None)
              
         train_len_list = []
         for x_train in x_train_list:
             train_len_list.append( x_train.shape[0])
         train_min_len = min(train_len_list)
        
         valid_len_list = []
         for x_valid in x_valid_list:
             valid_len_list.append( x_valid.shape[0])
         valid_min_len = min(valid_len_list)
        
         
         for i in range(len(x_train_list)):
             x_train = x_train_list[i][0:train_min_len,:]
             x_valid = x_valid_list[i][0:valid_min_len,:]
             y_train = y_train_list[i][0:train_min_len]
             y_valid = y_valid_list[i][0:valid_min_len]
             x_train_list[i] = x_train
             x_valid_list[i] = x_valid
             y_train_list[i] = y_train
             y_valid_list[i] = y_valid
             
         return x_train_list,x_valid_list,y_train_list,y_valid_list
    
    @staticmethod
    def tpot_select_model(x_train,y_train,x_test,y_test):
        from tpot import TPOTClassifier
        
        # create instance 
        tpot = TPOTClassifier(generations=10, population_size=50, verbosity=2, n_jobs=-1)
        # fit instance
        tpot.fit(x_train, y_train)        
        # evaluate performance on test data
        print(tpot.score(x_test, y_test))       
        
        # export the script used to create the best model
        tpot.export('tpot_exported_pipeline.py')
        
    
    @staticmethod
    def tpot_train_model(x_train_valid,x_test,y_train_valid,y_test,is_prob):
        import tpot_exported_pipeline as tep
        
        clf = tep.exported_pipeline
        clf.fit(x_train_valid,y_train_valid)  
        
        if is_prob:
            pred_test = clf.predict_proba(x_test)
        else:
            pred_test = clf.predict(x_test)
        pred_test = clf.predict(x_test)
            
        return clf,pred_test
    
    
    @staticmethod  
    def compute_prob_accuracy(pred_test,y_test,info=''):
        '''pred_test: a matrix, predicted probability of multi class'''
        right_count = 0
        for i in range(pred_test.shape[0]):
            l = pred_test[i,:].tolist()
            pred = l.index(max(l)) 
            if pred == y_test[i]:
                right_count += 1
        accuracy = right_count/len(y_test)
        log.log_and_print('%s accuracy : %.2f'%(info,accuracy))
        return accuracy
    
    @staticmethod  
    def compute_accuracy(pred_test,y_test,info=''):
        '''pred_test: a series, predicted values'''
        right_count = 0
        for i in range(pred_test.shape[0]): 
            if pred_test[i] == y_test[i]:
                right_count += 1
        accuracy = right_count/len(y_test)
        log.log_and_print('%s accuracy : %.2f'%(info,accuracy))
        return accuracy
    
    
    
class StackModel:
    
    def __init__(self):
        pass
    
    @staticmethod
    def layer1(X, Y, models, k_fold, is_step_gridsearch=True):
        
        layer1_list = []
        # =========first : cross-validation=======
        x_train_list,x_valid_list,y_train_list,y_valid_list = TrainModel.get_k_fold_sample(X, Y, k_fold)
        
        # =========second : layer1 construct ==========
        
        for i in range(k_fold):
            
            x_train = x_train_list[i]
            x_valid = x_valid_list[i]
            y_train = y_train_list[i]
            y_valid = y_valid_list[i]
              
            # try models
            model_list = []
            for j in range(len(models)):    
               
                model_obj = TrainModel.get_model_object(models[j], x_train, y_train, x_valid)
                #for lightGBM ,no grid search
                if models[j]=='lgb':
                    best_estimator = model_obj.fit(x_train, y_train, x_valid, y_valid, TrainModel.get_parameters(models[j], is_get_group=False))
                else:
                    clf = model_obj.get_clf( TrainModel.get_parameters(models[j]))
                    if is_step_gridsearch:
                        # step by step grid search
                        group_paras = TrainModel.get_parameters(models[j], is_get_group=True)
                        turned_paras = {}
                        for key in group_paras:
                            turned_paras[key] = group_paras[key]
                            best_estimator,best_parameters = TrainModel.grid_search_cv(clf, x_train, y_train, turned_paras)
                            turned_paras[key] = [best_parameters[key]]
                    else:
                        best_estimator,best_parameters = TrainModel.grid_search_cv(clf, x_train, y_train,  TrainModel.get_parameters(models[j], is_get_group=True))
                 
                model_list.append([model_obj,best_estimator])
                
                #valid
                pred_valid = model_obj.predict(best_estimator,x_valid)                
                #横向join模型输出
                if j==0:
                    pred_stack = pred_valid.copy()                     
                else:
                    pred_stack = np.hstack((pred_stack,pred_valid))  
            # 纵向concate样本
            if i==0:
                k_stack = pred_stack.copy()   
                y_real = y_valid.copy()  
            else:
                k_stack = np.vstack((k_stack,pred_stack))   
                y_real = np.hstack((y_real,y_valid)) 
            
            layer1_list.append(model_list)
            
        return layer1_list, k_stack, y_real
    
    
    @staticmethod
    def layer2(ensemble_model, ensem_x_train, ensem_y_train):
        ensemble_obj = TrainModel.get_model_object(ensemble_model,ensem_x_train,ensem_y_train, None)
        clf = ensemble_obj.get_clf( TrainModel.get_parameters(ensemble_model))
        best_estimator,best_parameters = TrainModel.grid_search_cv(clf, ensem_x_train, ensem_y_train,  TrainModel.get_parameters(ensemble_model, is_get_group=True))
        layer2_list = [ensemble_obj,best_estimator]
        return layer2_list
    
    
    @staticmethod
    def predict(layer1_list, layer2_list, x_test):
        # for each kfold and each model,compute test
        pred_stack_test_list = []
        for i in range(len(layer1_list)):
            for j in range(len(layer1_list[0])):    
                pred_test = layer1_list[i][j][0].predict( layer1_list[i][j][1], x_test)
                #横向join模型输出
                if j==0:                    
                    pred_stack_test = pred_test.copy()
                else:                    
                    pred_stack_test = np.hstack((pred_stack_test,pred_test))  
            pred_stack_test_list.append(pred_stack_test)    
        # k_fold mean
        sum = 0
        for k in range(len(layer1_list)-1):
            sum = pred_stack_test_list[k] + pred_stack_test_list[k+1]
        avg = sum/len(layer1_list)
        
        # predict
        pred = layer2_list[0].predict(layer2_list[1], avg)
        
        return pred
    
    
       