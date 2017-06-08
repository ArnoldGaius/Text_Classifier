#coding:utf-8
'''
Created on 2017.6.5

@author: Arnold_Gaius,ChiangClaire
'''
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import cross_validation
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd
import datetime
from sklearn.metrics.classification import confusion_matrix
from numpy import matrix

__all__ =  ['TextClassifier']

class ClassifierException(Exception):
    pass


class ClassifierNotTrainException(ClassifierException):
    def __init__(self):
        self.message = 'Text model has not been trained.'
        
class classifier_container:
    def __init__(self,model=None,original_data=None,vectorizer=None):
        self.model = model
        self.original_data = original_data
        self.vectorizer = vectorizer
    
    def load_Data(self,File_Path):
        print 'Loading File...'
        original_data = pd.read_csv(File_Path,header=None,encoding='utf-8',names=['Categorization','Text'])
        self.original_data = original_data
        print 'Loading File Finished !'
        return original_data
    
    def construct_w2v_vectorizer(self,X_train,Y_train):
        vectorizer = TfidfVectorizer(smooth_idf=True, sublinear_tf=True,use_idf=True,norm='l1')
        vectorizer.fit(X_train,Y_train)
        self.vectorizer = vectorizer
        return vectorizer
    
    def train(self,X_train=None, Y_train=None):
        if X_train == None:
            X_train = self.original_data['Text']
        if Y_train == None:    
            Y_train = self.original_data['Categorization']
        model = MultinomialNB(alpha=0.1,fit_prior=False)
        self.construct_w2v_vectorizer(X_train,Y_train)
        X_train_vec = self.vectorizer.transform(X_train)
        model.fit(X_train_vec, Y_train)
        self.model = model
        return model
    
    def Accuracy(self,X, Y):
        print 'Accuracy:'
        X_vec = self.vectorizer.transform(X)
        print self.model.score(X_vec, Y)
        
    def predict(self,X_test):
        pd.Series(X_test)
        X_test_vec = self.vectorizer.transform(X_test)
        return self.model.predict(X_test_vec)
    
    def confusion_matrix(self,Y_test,Y_predict):
        pd.set_option('display.height',1000)
        pd.set_option('display.max_rows',500)
        pd.set_option('display.max_columns',500)
        pd.set_option('display.width',1000)
        print 'Confusion Matrix :'
        labels = Y_test.unique()
        cm = metrics.confusion_matrix(Y_test,Y_predict,labels=labels)
        print pd.DataFrame(cm,columns = labels,index = labels)
        
    def plot_display(self,Y_test,Y_predict):
        print 'Plot display...'
        list_y_test = Y_test.value_counts().sort_index()
        list_y_predict = Y_predict.value_counts().sort_index()
        test_predict_count_df = pd.concat([list_y_test,list_y_predict,list_y_predict-list_y_test,abs(list_y_test-list_y_predict)],axis=1,keys=['Test count:','Predict count:','Sub Result:','Sub_Abs Result:'])
        print test_predict_count_df
        
        fig, ax = plt.subplots()  
        index = np.arange(len(self.original_data['Categorization'].unique()))  
        bar_width = 0.35
          
        opacity = 0.4
        rects1 = plt.bar(index, list_y_test, bar_width,alpha=opacity, color='b',label='test')  
        rects2 = plt.bar(index + bar_width, list_y_predict, bar_width,alpha=opacity,color='r',label='predict') 
        plt.xlabel('Group')
        plt.ylabel('Value')
        plt.title('Value by group')  
        plt.xticks(index + bar_width, (list_y_predict.index),rotation=-30)  
        plt.ylim(0,test_predict_count_df.values.max()*1.1)  
        plt.legend()  
        
        plt.tight_layout()  
        plt.show()
    
