# import needed libraries

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import  LinearSVC


# class for multinomial navie bayes
class MNB():
    
    # initialize model
    def  __init__(self ) -> None:
        
        self.model = MultinomialNB()
        
    # use for train model
    def  Fit(self , xTrain , yTrain) -> None: 
        
        self.model.fit(xTrain , yTrain)
        return None        
        
    # get the result of model
    def PredictProbility(self , sample) :
        
        return self.model.predict_proba(sample)
    
    def Prideict(self , sample) :
        return self.model.predict(sample)
    
    
class SVM():
    
    # initialize  model
    def __init__(self) :
        
        self.model = LinearSVC()
    
    # train the result
    def Fit(self , xTrain , yTrain) :
        
        self.model.fit(xTrain , yTrain)
        return None
    
    # probility of each class for sample
    def PredictProbility(self , sample):
        
        return self.model._predict_proba_lr(sample)
    
    # predicred class of sample
    def Prideict(self , sample ) :
        return self.model.predict(sample)
    
