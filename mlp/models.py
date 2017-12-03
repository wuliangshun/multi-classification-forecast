# -*- coding: utf-8 -*-
"""
models
"""

import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from xgboost.sklearn import XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier



class ModelBase(object):
    """
        Base
    """
    def __init__(self ,x_tr, y_tr, x_te):
       
        self.x_train = x_tr
        self.y_train = y_tr
        self.x_test = x_te
        
        pass
    
        
    @staticmethod
    def get_clf(parameters):

        print('This Is Base Model!')
        clf = DecisionTreeClassifier()

        return clf

    @staticmethod
    def start_and_get_model_name():

        
        print('This Is Base Model!')
        

        model_name = 'base'

        return model_name

    def fit(self, x_train, y_train, x_valid, y_valid, parameters):

        # Get Classifier
        clf = self.get_clf(parameters)
        
        # Training Model
        clf.fit(x_train, y_train)

        return clf
    
    def get_importance(self, clf):

        
        print('Feature Importance : '+self.start_and_get_model_name())

        self.importance = clf.feature_importances_
        self.indices = np.argsort(self.importance)[::-1]

        feature_num = len(self.importance)

        for f in range(feature_num):
            print("%d | feature %d | %d" % (f + 1, self.indices[f], self.importance[self.indices[f]]))
            
        return self.importance
    
    def predict(self, clf, x_test, is_prob=True):

        if is_prob:
            prob_test = np.array(clf.predict_proba(x_test))
        else:
            prob_test = np.array(clf.predict(x_test))
        
        return prob_test
   


class LRegression(ModelBase):
    """
        Logistic Regression
    """
    @staticmethod
    def get_clf(parameters):

        
        clf = LogisticRegression(**parameters)

        return clf

    @staticmethod
    def start_and_get_model_name():

        
        print('Training Logistic Regression...')
        

        model_name = 'lr'

        return model_name

    def get_importance(self, clf):
        
        print('Feature Importance : '+self.start_and_get_model_name())
        self.importance = np.abs(clf.coef_)[0]
        indices = np.argsort(self.importance)[::-1]

        feature_num = self.x_train.shape[1]

        for f in range(feature_num):
            print("%d | feature %d | %f" % (f + 1, indices[f], self.importance[indices[f]]))
        
        return self.importance

class KNearestNeighbor(ModelBase):
    """
        k-Nearest Neighbor Classifier
    """
    @staticmethod
    def get_clf(parameters):

        
        clf = KNeighborsClassifier(**parameters)

        return clf

    @staticmethod
    def start_and_get_model_name():

        
        print('Training k-Nearest Neighbor Classifier...')
        

        model_name = 'knn'

        return model_name


class SupportVectorClustering(ModelBase):
    """
        SVM - Support Vector Clustering
    """
    @staticmethod
    def get_clf(parameters):

        
        clf = SVC(**parameters)

        return clf

    @staticmethod
    def start_and_get_model_name():

        
        print('Training Support Vector Clustering...')
        

        model_name = 'svm'

        return model_name


class Gaussian(ModelBase):
    """
        Gaussian NB
    """
    @staticmethod
    def get_clf(parameters):

        
        clf = GaussianNB(**parameters)

        return clf

    @staticmethod
    def start_and_get_model_name():

        
        print('Training Gaussian...')
        

        model_name = 'gs'

        return model_name


class DecisionTree(ModelBase):
    """
        Decision Tree
    """
    @staticmethod
    def get_clf(parameters):

        
        clf = DecisionTreeClassifier(**parameters)

        return clf

    @staticmethod
    def start_and_get_model_name():

        
        print('Training Decision Tree...')
        

        model_name = 'dt'

        return model_name


class RandomForest(ModelBase):
    """
        Random Forest
    """
    @staticmethod
    def get_clf(parameters):

        
        clf = RandomForestClassifier(**parameters)

        return clf

    @staticmethod
    def start_and_get_model_name():

        
        print('Training Random Forest...')
        

        model_name = 'rf'

        return model_name


class ExtraTrees(ModelBase):
    """
        Extra Trees
    """
    @staticmethod
    def get_clf(parameters):

        
        clf = ExtraTreesClassifier(**parameters)

        return clf

    @staticmethod
    def start_and_get_model_name():

        
        print('Training Extra Trees...')
        

        model_name = 'et'

        return model_name


class AdaBoost(ModelBase):
    """
        AdaBoost
    """
    @staticmethod
    def get_clf(parameters):

        
        clf = AdaBoostClassifier(**parameters)

        return clf

    @staticmethod
    def start_and_get_model_name():

        
        print('Training AdaBoost...')
        

        model_name = 'ab'

        return model_name


class GradientBoosting(ModelBase):
    """
        Gradient Boosting
    """
    @staticmethod
    def get_clf(parameters):

        
        clf = GradientBoostingClassifier(**parameters)

        return clf

    @staticmethod
    def start_and_get_model_name():

        
        print('Training Gradient Boosting...')
        

        model_name = 'gb'

        return model_name  
    
class SKLearnXGBoost(ModelBase):
    """
        XGBoost using sklearn module
    """
    @staticmethod
    def get_clf(parameters=None):

        clf = XGBClassifier(**parameters)

        return clf
    
    @staticmethod
    def start_and_get_model_name():

        
        print('Training XGBoost(sklearn)...')
        

        model_name = 'xgb_sk'

        return model_name
    
    def fit( self, x_train, y_train, x_valid, y_valid, parameters):
        
        # Get Classifier
        clf = self.get_clf(parameters)

        # Training Model
        clf.fit(x_train, y_train, 
                eval_set=[(x_train, y_train), (x_valid, y_valid)],
                early_stopping_rounds=100, eval_metric='logloss', verbose=True)

        return clf
    
    def get_importance(self, clf):
        
        print('Feature Importance : '+self.start_and_get_model_name())

        self.importance = clf.feature_importances_
        self.indices = np.argsort(self.importance)[::-1]

        feature_num = len(self.importance)

        for f in range(feature_num):
            print("%d | feature %d | %f" % (f + 1, self.indices[f], self.importance[self.indices[f]]))
        
        return self.importance
    
    def get_accurancy(y_true, y_pred):
         return metrics.accuracy_score(y_true, y_pred)
     
class SKLearnLightGBM(ModelBase):
    """
        LightGBM using sklearn module
    """
    @staticmethod
    def get_clf(parameters=None):

        clf = LGBMClassifier(**parameters)

        return clf

    @staticmethod
    def start_and_get_model_name():
        
        print('Training LightGBM(sklearn)...')        

        model_name = 'lgb_sk'

        return model_name

    @staticmethod
    def fit(self, x_train, y_train, x_valid, y_valid, parameters):

        # Get Classifier
        clf = self.get_clf(parameters)

        idx_category = [x_train.shape[1] - 1]
        print('Index of categorical feature: {}'.format(idx_category))

        # Fitting and Training Model
        clf.fit(x_train, y_train,  categorical_feature=idx_category,
                eval_set=[(x_train, y_train), (x_valid, y_valid)], eval_names=['train', 'eval'],
                early_stopping_rounds=100, 
                eval_metric='logloss', verbose=True)

        return clf
    
class LightGBM(ModelBase):
    """
        LightGBM
    """
    def __init__(self, x_tr, y_tr, x_te, num_boost_round):
        super(LightGBM, self).__init__(x_tr, y_tr, x_te)

        self.num_boost_round = num_boost_round
        
    @staticmethod
    def start_and_get_model_name():

        
        print('Training LightGBM...')
        

        model_name = 'lgb'

        return model_name
    
    def fit(self, x_train, y_train, x_valid, y_valid, parameters):

        # Create Dataset
        idx_category = [x_train.shape[1] - 1]
        print('Index of categorical feature: {}'.format(idx_category))
        

        d_train = lgb.Dataset(x_train, label=y_train,  categorical_feature=idx_category)
        d_valid = lgb.Dataset(x_valid, label=y_valid,  categorical_feature=idx_category)

        # Booster
        bst = lgb.train(parameters, d_train, num_boost_round=self.num_boost_round,
                        valid_sets=[d_valid, d_train], valid_names=['Valid', 'Train'])

        return bst
    
    def get_importance(self, bst):

        
        print('Feature Importance : '+self.start_and_get_model_name())

        self.importance = bst.feature_importance()
        self.indices = np.argsort(self.importance)[::-1]

        feature_num = len(self.importance)

        for f in range(feature_num):
            print("%d | feature %d | %d" % (f + 1, self.indices[f], self.importance[self.indices[f]]))

        return self.importance
        
        
    def predict(self, bst, x_test, is_prob=True):

        prob_test = bst.predict(x_test)

        return prob_test


class CatBoost(ModelBase):
    """
        CatBoost
    """
    @staticmethod
    def get_clf(parameters=None):

        
        clf = CatBoostClassifier(**parameters)

        return clf

    @staticmethod
    def start_and_get_model_name():

        
        print('Training CatBoost...')
        

        model_name = 'cb'

        return model_name

    def get_importance(self, clf):

        
        print('Feature Importance : '+self.start_and_get_model_name())

        self.importance = clf.feature_importances_
        self.indices = np.argsort(self.importance)[::-1]

        feature_num = len(self.importance)

        for f in range(feature_num):
            print("%d | feature %d | %d" % (f + 1, self.indices[f], self.importance[self.indices[f]]))
            
        return self.importance

    def fit(self, x_train, y_train, x_valid, y_valid, parameters):

        # Get Classifier
        clf = self.get_clf(parameters)

        idx_category = [x_train.shape[1] - 1]
        print('Index of categorical feature: {}'.format(idx_category))

        # Fitting and Training Model
        clf.fit(X=x_train, y=y_train, cat_features=idx_category, 
                baseline=None, use_best_model=None, eval_set=(x_valid, y_valid), verbose=True, plot=False)

        return clf

    
class DeepNeuralNetworks(ModelBase):
    """
        Deep Neural Networks
    """
    
    def __init__(self, x_tr, y_tr, x_te, y_te, parameters):
        
        super(DeepNeuralNetworks,self).__init__(x_tr, y_tr, x_te)
        
        self.y_test = y_te
       
        
        # Hyperparameters
        self.parameters = parameters
        self.epochs = parameters['epochs']
        self.unit_number = parameters['unit_number']
        self.n_classes = parameters['n_classes']
        self.learning_rate = parameters['learning_rate']
        self.learning_rate_decay = parameters['learning_rate_decay']
        self.regularization_rate = parameters['regularization_rate']
        self.batch_size = parameters['batch_size']
        self.display_step = parameters['display_step']
        
    @staticmethod
    def start_and_get_model_name():
        
        print('Training Deep Neural Networks...')        

        model_name = 'dnn'

        return model_name   

    def add_layer(self, inputs, in_size, out_size, activation_function=None):
        # add one more layer and return the output of this layer
        Weights = tf.Variable(tf.random_normal([in_size, out_size]))
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs,Weights

    def train(self):  
        
        feature_num = self.x_train.shape[1]
        total_batch = int(self.x_train.shape[0]/self.batch_size)
        
        x = tf.placeholder(tf.float32, [None, feature_num]) # 用placeholder先占地方，样本个数不确定为None
        y = tf.placeholder(tf.float32, [None, self.n_classes]) # 用placeholder先占地方，样本个数不确定为None

        
        hidden,weights1 = self.add_layer(x, feature_num, self.unit_number, activation_function=tf.nn.relu)
        outputs,weights2 = self.add_layer(hidden, self.unit_number, self.n_classes) #, activation_function=tf.nn.softmax
       
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=y))
        regularizer = tf.contrib.layers.l2_regularizer(self.regularization_rate)        
        regularization = regularizer(weights1) + regularizer(weights2)
        loss = cross_entropy + regularization
        
        global_step = tf.Variable(0, trainable=False)
        
        learning_rate = tf.train.exponential_decay(self.learning_rate,
                                                   global_step,
                                                   total_batch,
                                                   self.learning_rate_decay)
        
        optm = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
        
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(outputs,1), tf.argmax(y,1)),tf.float32))

        init = tf.global_variables_initializer()
        
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(self.epochs):
                avg_loss = 0.
                for i in range(total_batch):
                    batch_xs = self.x_train[(i)*self.batch_size:(i+1)*self.batch_size,:]
                    batch_ys = self.y_train[(i)*self.batch_size:(i+1)*self.batch_size,:]     
                    
                    sess.run(optm, feed_dict={x: batch_xs, y: batch_ys})
                    avg_loss += sess.run(loss, feed_dict={x: batch_xs, y: batch_ys})/total_batch
                if epoch%self.display_step == 0:
                    train_accurancy = sess.run(accuracy,feed_dict={x: batch_xs, y: batch_ys})
                    test_accurancy = sess.run(accuracy,feed_dict={x: self.x_test, y: self.y_test})
                    print("Epoch:%d/%d,\tloss:%.9f\ttrain_accurancy:%.3f\ttest_accurancy:%.3f" %(epoch,self.epochs,avg_loss,train_accurancy,test_accurancy))
