#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 20:26:49 2019

@author: dauku

BernoulliNB with changable Tc and Phs
"""

# In[1]:
"""import packages"""

import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate



# In[2]:
List = []
for i in range(50):
    Indicator = []
    HclassList = []
    UncertainList = []
    LclassList = []
    HclassList.append(np.nan)
    UncertainList.append(np.nan)
    LclassList.append(np.nan)
    ExpensiveHList = []
    ExpensiveLList = []
    RecallList = []
    PrecisionList = []
    F1List = []
    PhsList = []
    TcList = []
    TcIndicator = []
    TcIndicator.append(np.nan)
    
    
    # In[3]:
    
    
    items = 20  # number of bags
    combinations = list(itertools.product([0, 1], repeat = items))  # create all the combinations of choices 
    candidates = np.asarray(combinations)  # create a array that contains all the combinations 
    #candidates.shape
    
    weights = [94, 70, 90, 97, 54, 31, 82, 97, 1, 58, 96, 96, 87, 53, 62, 89, 68, 58, 81, 83]  # weight of items
    values = [3, 41, 22, 30, 45, 99, 75, 76, 79, 77, 41, 98, 31, 28, 58, 32, 99, 48, 20, 3]  # value of items 
    variables = np.asarray([weights, values])  # create a array that contains weight and value of items
    TotalWeight = sum(weights)/2  # Constraint of weight
    
    Ntr = 100  # number of items in the initial expensive evaluation
    Ns = 50  # number of items in the sequential expensive evaluation
    Phs = 0.9  # Initial porportion of high-class points 
    PhsList.append(Phs)
    
    
    """Create the DataFrame"""
    Data = pd.DataFrame(candidates, columns = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13',
                             'X14', 'X15', 'X16', 'X17', 'X18', 'X19', 'X20'])
    Data['Obj'] = np.nan
    Data['Weight'] = np.nan
    Data['Label'] = np.nan
    Data['Plow'] = np.nan
    Data['Phigh'] = np.nan
    Data['PredictLabel'] = np.nan
    Data.head()
    
    # %% 
    """Calculate the true global optimum"""
    Objs = Data.loc[:, 'X1':'X20']*variables[1]
    ObjValues = Objs.sum(axis = 1).values
    Data.loc[:, 'Obj'] = ObjValues
    WeightCals = Data.loc[:, 'X1':'X20']*variables[0]
    Weights = WeightCals.sum(axis = 1).values
    Data.loc[:, 'Weight'] = Weights
    WWIndices = Data[Data.Weight <= TotalWeight].index
    TrueOptimum = Data.loc[WWIndices, 'Obj'].max()
    
    # %%
    """find out the number of points which satisfies the constrain"""
    SatisfyConstrain = Data.loc[WWIndices].sort_values(by = ['Obj'], ascending = False)
    GreaterThan700 = SatisfyConstrain[SatisfyConstrain.Obj >= 700]
    Data['Obj'] = np.nan
    Data['Weight'] = np.nan
    
    
    # In[4]:
    
    
    """Conduct the initial expensive evaluation"""
    InitialIndices = np.random.choice(Data.index.values, Ntr, replace = False)  # Randomly pick 100 design choices for the initial expensive evaluation
    Objs = Data.loc[InitialIndices, 'X1':'X20']*variables[1]
    ObjValues = Objs.sum(axis = 1).values
    Data.loc[InitialIndices, 'Obj'] = ObjValues
    WeightCals = Data.loc[InitialIndices, 'X1':'X20']*variables[0]
    Weights = WeightCals.sum(axis = 1).values
    Data.loc[InitialIndices, 'Weight'] = Weights
    WWIndices = Data.loc[InitialIndices][Data.Weight.loc[InitialIndices] <= TotalWeight].index
    Optimum = Data.loc[WWIndices, 'Obj'].max()
    Optimum
    OptimumList = []
    OptimumList.append(Optimum)
    
    """Assign Labels to initial indices"""
    TcPercent = int(Ntr*0.05)
    SortedData = Data.loc[WWIndices].sort_values(by = ['Obj'], ascending = False)
    Tc0 = SortedData.iloc[TcPercent-1,-6]
    Tc = Tc0 - 5
    TcList.append(Tc0)
    
    
    def AssignLabel(design):
        if design.Obj >= Tc0 and design.Weight <= TotalWeight:
            return(1)
        else:
            return(0)
    Data.loc[InitialIndices, 'Label'] = Data.loc[InitialIndices,:].apply(AssignLabel, axis = 1)
    HNumber = len(Data.loc[InitialIndices][Data.loc[InitialIndices, 'Label'] == 1])
    LNumber = len(Data.loc[InitialIndices][Data.loc[InitialIndices, 'Label'] == 0])
    ExpensiveHList.append(HNumber)
    ExpensiveLList.append(LNumber)
    
    
    
    # In[8]:
    
    #"""Initial assess of the classifier"""
    #x = Data.loc[InitialIndices, "X1":"X20"]
    #y = Data.loc[InitialIndices, "Label"]
    #model = BernoulliNB()
    #metrics = ['recall', 'precision', 'f1']
    #scores = cross_val_score(model, x, y, cv=5, scoring='precision')
    #print(scores)
    #print(scores.mean())
    #RecallList.append(scores.mean())
    
    # In[9]:
    
    
    """Cross Validation"""
    x = Data.loc[InitialIndices, "X1":"X20"]
    y = Data.loc[InitialIndices, "Label"]
    
    kf = KFold(n_splits = 5)
    Confusion = pd.DataFrame(columns = ["NN", "NP", "PN", "PP"])
    
    for train_index, test_index in kf.split(x):
        x_train_10 = x.iloc[train_index, :]
        y_train_10 = y.iloc[train_index]
        x_test_10 = x.iloc[test_index, :]
        y_test_10 = y.iloc[test_index]
        model = MultinomialNB()
        model.fit(x_train_10, y_train_10)
        a = model.predict(x_test_10)
        result = pd.DataFrame({'True': y_test_10, 'Predicted': a})
        CM = confusion_matrix(result['True'], result['Predicted'], labels = [0, 1])
        Confusion = Confusion.append(pd.Series([CM[0,0],CM[0,1],CM[1,0],CM[1,1]], index = ["NN", "NP", "PN", "PP"]), ignore_index = True)
    
    # Metrics
    def Recall(ConfusionMatrix):
        if (ConfusionMatrix.PN + ConfusionMatrix.PP) != 0:
            RecallScore = ConfusionMatrix.PP / (ConfusionMatrix.PN + ConfusionMatrix.PP)
            return(RecallScore)
        else:
            return(np.nan)
    def Precision(ConfusionMatrix):
        if (ConfusionMatrix.NP + ConfusionMatrix.PP) != 0:
            Precision = ConfusionMatrix.PP / (ConfusionMatrix.NP + ConfusionMatrix.PP)
            return(Precision)
        else:
            return(np.nan)
            
    Confusion['Recall'] = Confusion.apply(Recall, axis = 1)
    Confusion['Precision'] = Confusion.apply(Precision, axis = 1)
    
    def f1(ConfusionMatrix):
        if ConfusionMatrix.Recall > 0 and ConfusionMatrix.Precision > 0:
            f1score = 2*ConfusionMatrix.Recall*ConfusionMatrix.Precision/(ConfusionMatrix.Recall+ConfusionMatrix.Precision)
            return(f1score)
        else:
            return(np.nan)
    Confusion['F1'] = Confusion.apply(f1, axis = 1)
    Scores = Confusion.loc[:, 'Recall':'F1'].mean(axis = 0)
    RecallList.append(Scores[0])
    PrecisionList.append(Scores[1])
    F1List.append(Scores[2])
    
    #From the above result, we can see that the recall score (True positive rate is very low)
    
    # In[11]:
    
    
    """Initiate NB classifier to generate cheap points"""
    model = MultinomialNB()
    featureTrain = Data.loc[InitialIndices, 'X1':'X20'].values
    labelTrain = Data.loc[InitialIndices, 'Label'].values
    model.fit(featureTrain, labelTrain)
    
    """Generate Cheap points"""
    ExcludedIndices = np.setdiff1d(np.array(Data.index), InitialIndices)
    CheapPoint = model.predict(Data.loc[ExcludedIndices, 'X1':'X20'])
    Prob = model.predict_proba(Data.loc[ExcludedIndices, 'X1':'X20'])
    Data.loc[ExcludedIndices, 'PredictLabel'] = CheapPoint
    Data.loc[ExcludedIndices, 'Plow':'Phigh'] = Prob
    
    Indicator.append(np.nan)
    
    
    # In[12]:
    
    
    """Looping repeat until meet the termination criteria"""
    for i in range(22):
        
        # Binning cheap points into three categories: HHpoint, Uncertain point, LHpoint
        Uncertain = Data[(Data.Plow >= 0.4) & (Data.Plow <= 0.6)]
        UncertainList.append(len(Uncertain))
        HclassH = Data[(Data.PredictLabel == 1) & (Data.Phigh > 0.6)]
        HclassList.append(len(HclassH))
        LclassH = Data[(Data.PredictLabel == 0) & (Data.Plow > 0.6)]
        LclassList.append(len(LclassH))
        
        # Phs update 
        if i >= 2:
            if OptimumList[i] <= OptimumList[i-2] and OptimumList[i-1] <= OptimumList[i-2]:
                if Phs == 0.9:
                    Phs = 0.5
            else:
                Phs = Phs
        HHPhs = int(Ns*Phs)
        
        # select points for sequential expensive evaluations 
        if (len(Uncertain) + len(HclassH)) < Ns:
            HHIndices = HclassH.index
            UnIndices = Uncertain.index
            LHIndices = np.random.choice(LclassH.index, Ns - len(HHIndices) - len(UnIndices), replace = False)
            NsIndices = np.concatenate((HHIndices, UnIndices, LHIndices), axis = None)
            Indicator.append('Less')
        else:
            if len(HclassH) >= HHPhs:
                HHIndices = np.random.choice(HclassH.index, HHPhs, replace = False)
                UnIndices = np.random.choice(Uncertain.index, Ns - HHPhs)
                NsIndices = np.concatenate((HHIndices, UnIndices), axis = None)
                Indicator.append('HH')
            else:
                HHIndices = HclassH.index
                UnIndices = np.random.choice((Uncertain.index), Ns - len(HHIndices), replace = False)
                NsIndices = np.concatenate((HHIndices, UnIndices), axis = None)
                Indicator.append('H')
        
        PhsList.append(Phs)
        if Phs < 0.9:
            Phs = round(Phs + 0.08, 2)
        
        Data.loc[NsIndices, 'Plow':'Phigh'] = np.nan
        
        # expensive evaluation
        Objs = Data.loc[NsIndices, 'X1':'X20']*variables[1]
        ObjValues = Objs.sum(axis = 1).values
        Data.loc[NsIndices, 'Obj'] = ObjValues
        WeightCals = Data.loc[NsIndices, 'X1':'X20']*variables[0]
        Weights = WeightCals.sum(axis = 1).values
        Data.loc[NsIndices, 'Weight'] = Weights
        WWIndices = Data.loc[NsIndices][Data.Weight.loc[NsIndices] <= TotalWeight].index
        Optimum = Data.loc[WWIndices, 'Obj'].max()
        OptimumList.append(Optimum)
        
        if len(HclassH) > Ns:
            if Tc <= max(OptimumList):
                if Tc == max(OptimumList):
                    Tc = Tc - 5
                Tc = Tc - 0.5*(Tc - max(OptimumList))
            else:
                Tc = max(OptimumList)
            TcIndicator.append('G')
        elif len(HclassH) <= 0.1*Ns:
            if Tc >= Tc0:
                if Tc == Tc0:
                    Tc = Tc + 5
                Tc = Tc + 0.05*(Tc0 - Tc)
            else:
                Tc = Tc0
            TcIndicator.append('L')
        else:
            Tc = Tc
            TcIndicator.append('S')
        TcList.append(Tc)
        
        def AssignLabel2(design):
            if design.Obj >= Tc and design.Weight <= TotalWeight:
                return(1)
            else:
                return(0)
        Data.loc[NsIndices, 'Label'] = Data.loc[NsIndices,:].apply(AssignLabel2, axis = 1)
        
        HNumber += len(Data.loc[NsIndices][Data.loc[NsIndices, 'Label'] == 1])
        LNumber += len(Data.loc[NsIndices][Data.loc[NsIndices, 'Label'] == 0])
        ExpensiveHList.append(HNumber)
        ExpensiveLList.append(LNumber)
    
        # Retrain the model and generate cheap points
        NsIndices = np.concatenate((InitialIndices, NsIndices), axis = None)
        print(len(NsIndices))
        featureTrain = Data.loc[NsIndices, 'X1':'X20'].values
        labelTrain = Data.loc[NsIndices, 'Label'].values
        model.fit(featureTrain, labelTrain)
        ExcludedIndices = np.setdiff1d(np.array(Data.index), NsIndices)
        CheapPoint = model.predict(Data.loc[ExcludedIndices, 'X1':'X20'])
        Prob = model.predict_proba(Data.loc[ExcludedIndices, 'X1':'X20'])
        Data.loc[ExcludedIndices, 'PredictLabel'] = CheapPoint
        Data.loc[ExcludedIndices, 'Plow':'Phigh'] = Prob
    
        InitialIndices = NsIndices
    
        # Examine the metrics of the classifier for each iteration
        x = Data.loc[NsIndices, 'X1':'X20']    
        y = Data.loc[NsIndices, 'Label']
        
        kf = KFold(n_splits = 5)
        Confusion = pd.DataFrame(columns = ["NN", "NP", "PN", "PP"])
        
        for train_index, test_index in kf.split(x):
            x_train_10 = x.iloc[train_index, :]
            y_train_10 = y.iloc[train_index]
            x_test_10 = x.iloc[test_index, :]
            y_test_10 = y.iloc[test_index]
            model = MultinomialNB()
            model.fit(x_train_10, y_train_10)
            a = model.predict(x_test_10)
            result = pd.DataFrame({'True': y_test_10, 'Predicted': a})
            CM = confusion_matrix(result['True'], result['Predicted'], labels = [0, 1])
            Confusion = Confusion.append(pd.Series([CM[0,0],CM[0,1],CM[1,0],CM[1,1]], index = ["NN", "NP", "PN", "PP"]), ignore_index = True)
        
        # Metrics
        Confusion['Recall'] = Confusion.apply(Recall, axis = 1)
        Confusion['Precision'] = Confusion.apply(Precision, axis = 1)
        Confusion['F1'] = Confusion.apply(f1, axis = 1)
        Scores = Confusion.loc[:, 'Recall':'F1'].mean(axis = 0)
        RecallList.append(Scores[0])
        PrecisionList.append(Scores[1])
        F1List.append(Scores[2])
        
    
    
    Process = np.array([OptimumList, Indicator, PhsList, TcList, TcIndicator, HclassList, UncertainList, LclassList, ExpensiveHList, ExpensiveLList, RecallList, PrecisionList, F1List])
    Process = np.transpose(Process)
    ProcessIndicators = pd.DataFrame(Process, columns = ['OptimumList', 'Indicator', 'PhsList', 'TcList', 'TcIndicator', 'HclassList', 'UncertainList', 'LclassList', 'ExpensiveHList', 'ExpensiveLList', 'RecallList', 'PrecisionList', 'F1List'])
    
    List.append(OptimumList)
    
ListArray = np.array(List)
Average = np.mean(ListArray, axis = 0).reshape(23, 1)
Std = np.std(ListArray, axis = 0).reshape(23, 1)
Percentile90 = np.percentile(ListArray, 90, axis = 0).reshape(23,1)
Percentile10 = np.percentile(ListArray, 10, axis = 0).reshape(23,1)
Evaluations = (np.arange(23)*50) + 100
Evaluations = Evaluations.reshape(23, 1)
PrintingData = np.concatenate((Evaluations, Average, Std, Percentile10, Percentile90), axis = 1)
PrintingDataFrame = pd.DataFrame(PrintingData, columns = ['Evaluations', 'Mean', 'Std', 'Percentile10', 'Percentile90'])
PrintingDataFrame.to_csv("/Users/dauku/Desktop/Python/MAE586/TwentyKnapsack/3-20/MultinomialPrintingDataFrame.csv", index = False)

PrintingRecall = pd.DataFrame(np.concatenate((Evaluations, np.array(RecallList).reshape(23, 1)), axis = 1), columns = ["Evaluations", "Recall"])
PrintingRecall.to_csv("/Users/dauku/Desktop/Python/MAE586/TwentyKnapsack/3-20/MultinomialProcessIndicators.csv", index = False)
