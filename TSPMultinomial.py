#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 21:04:43 2019

@author: dauku
"""

# In[1]:
"""impport package"""
import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


# In[2]:
"""Indicators"""
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
"""Create Data"""

# City coordinates
City = pd.DataFrame({"City": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                     "Abscissa": [14.8, 98.6, 5, 39.9, 28, 57, 26.2, 67.9, 76, 86.4, 47.3],
                     "Ordinate": [42.7, 29.8, 4.5, 12, 22.6, 79.2, 90.7, 55.1, 47.1, 67.5, 98.5]})

# Premutation
import itertools
Permutations = list(itertools.permutations([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
Permutations = np.asarray(Permutations)
StartingCity = np.zeros(3628800).reshape(3628800, 1)
permuMatrix = np.column_stack((StartingCity, Permutations, StartingCity))
Testing = permuMatrix[0:3, :]

# Calculate distance
def Distance(row):
    DisList = []
    for i in range(City.shape[0]):
        End = int(row[i+1])
        Starting = int(row[i])
        EndPoint = City.iloc[End, 1:3]
        StartingPoint = City.iloc[Starting, 1:3]
        Dis = np.linalg.norm(EndPoint - StartingPoint)
        DisList.append(Dis)
    return(sum(DisList))
Distances = np.apply_along_axis(Distance, 1, permuMatrix)

# Global Minimum
GlobalMinimum = np.amin(Distances)

# Combine matrices and build the dataframe
permuMatrix = np.column_stack((permuMatrix, Distances))
permuData = pd.DataFrame(permuMatrix, columns = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'Distance'])
permuData['Obj'] = np.nan
permuData['Label'] = np.nan
permuData['Plow'] = np.nan
permuData['Phigh'] = np.nan
permuData['PredictLabel'] = np.nan

# In[4]:
List = []
for i in range(50):
    """CGS"""
    
    """Specify CGS parameters"""
    Ntr = 100  # number of designs in the initial expensive evaluation
    Ns = 100  # number of designs in the sequential expensive evaluation
    Phs = 0.5  # Initial porportion of high-class points 
    PhsList.append(Phs)
    
    """Conduct the initial expensive evaluation"""
    InitialIndices = np.random.choice(permuData.index.values, Ntr, replace = False)  # Randomly pick 100 design choices for the initial expensive evaluation
    permuData.loc[InitialIndices, "Obj"] = permuData.loc[InitialIndices, 'X1':'X12'].apply(Distance, axis=1)
    
    
    Optimum = permuData.loc[InitialIndices, 'Obj'].min()
    Optimum
    OptimumList = []
    OptimumList.append(Optimum)
    
    """Assign labels to initial indices"""
    TcPercent = int(Ntr*0.05)
    SortedData = permuData.loc[InitialIndices].sort_values(by = ['Obj'], ascending = True)
    Tc0 = SortedData.Obj.iloc[TcPercent-1]
    Tc = Tc0 + 5
    TcList.append(Tc0)
    
    def AssignLabel(design):
            if design.Obj <= Tc0:
                return(1)
            else:
                return(0)
    permuData.loc[InitialIndices, 'Label'] = permuData.loc[InitialIndices,:].apply(AssignLabel, axis = 1)
    HNumber = len(permuData.loc[InitialIndices][permuData.loc[InitialIndices, 'Label'] == 1])
    LNumber = len(permuData.loc[InitialIndices][permuData.loc[InitialIndices, 'Label'] == 0])
    ExpensiveHList.append(HNumber)
    ExpensiveLList.append(LNumber)
    
    """KNN GridSearch"""
    knn_params = {"n_neighbors": list(range(1, 31))}
    
    """Cross Validation"""
    x = permuData.loc[InitialIndices, "X1":"X12"]
    y = permuData.loc[InitialIndices, "Label"]
    
    kf = KFold(n_splits = 5)
    Confusion = pd.DataFrame(columns = ["TN", "FP", "FN", "TP"])
    
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
        Confusion = Confusion.append(pd.Series([CM[0,0],CM[0,1],CM[1,0],CM[1,1]], index = ["TN", "FP", "FN", "TP"]), ignore_index = True)
    
    # Metrics
    def Recall(ConfusionMatrix):
        if (ConfusionMatrix.FN + ConfusionMatrix.TP) != 0:
            RecallScore = ConfusionMatrix.TP / (ConfusionMatrix.FN + ConfusionMatrix.TP)
            return(RecallScore)
        else:
            return(np.nan)
    def Precision(ConfusionMatrix):
        if (ConfusionMatrix.FP + ConfusionMatrix.TP) != 0:
            Precision = ConfusionMatrix.TP / (ConfusionMatrix.FP + ConfusionMatrix.TP)
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
    
    """Initiate NB classifier to generate cheap points"""
    model = MultinomialNB()
    featureTrain = permuData.loc[InitialIndices, 'X1':'X12'].values
    labelTrain = permuData.loc[InitialIndices, 'Label'].values
    model.fit(featureTrain, labelTrain)
    
    """Generate Cheap points"""
    ExcludedIndices = np.setdiff1d(np.array(permuData.index), InitialIndices)
    CheapPoint = model.predict(permuData.loc[ExcludedIndices, 'X1':'X12'])
    Prob = model.predict_proba(permuData.loc[ExcludedIndices, 'X1':'X12'])
    permuData.loc[ExcludedIndices, 'PredictLabel'] = CheapPoint
    permuData.loc[ExcludedIndices, 'Plow':'Phigh'] = Prob
    
    Indicator.append(np.nan)
    
    """Looping repeat until meet the termination criteria"""
    for i in range(79):
        
        # Binning cheap points into three categories: HHpoint, Uncertain point, LHpoint
        Uncertain = permuData[(permuData.Plow >= 0.4) & (permuData.Plow <= 0.6)]
        UncertainList.append(len(Uncertain))
        HclassH = permuData[(permuData.PredictLabel == 1) & (permuData.Phigh > 0.6)]
        HclassList.append(len(HclassH))
        LclassH = permuData[(permuData.PredictLabel == 0) & (permuData.Plow > 0.6)]
        LclassList.append(len(LclassH))
        
    #   Phs update 
        if i >= 2:
            if OptimumList[i] >= OptimumList[i-2] and OptimumList[i-1] >= OptimumList[i-2]:
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
        
        permuData.loc[NsIndices, 'Plow':'Phigh'] = np.nan
        
        
        permuData.loc[NsIndices, "Obj"] = permuData.loc[NsIndices, "X1":"X12"].apply(Distance, axis = 1)
        Optimum = permuData.loc[NsIndices, 'Obj'].min()
        OptimumList.append(Optimum)
        
        if len(HclassH) > Ns:
            if Tc >= min(OptimumList):
                if Tc == min(OptimumList):
                    Tc = Tc + 5
                Tc = Tc - 0.5*(Tc - min(OptimumList))
            else:
                Tc = min(OptimumList)
            TcIndicator.append('G')
        elif len(HclassH) <= 0.1*Ns:
            if Tc <= Tc0:
                if Tc == Tc0:
                    Tc = Tc - 5
                Tc = Tc + 0.05*(Tc0 - Tc)
            else:
                Tc = Tc0
            TcIndicator.append('L')
        else:
            Tc = Tc
            TcIndicator.append('S')
        TcList.append(Tc)
        
        def AssignLabel2(design):
            if design.Obj <= Tc:
                return(1)
            else:
                return(0)
        permuData.loc[NsIndices, 'Label'] = permuData.loc[NsIndices,:].apply(AssignLabel2, axis = 1)
        
        HNumber += len(permuData.loc[NsIndices][permuData.loc[NsIndices, 'Label'] == 1])
        LNumber += len(permuData.loc[NsIndices][permuData.loc[NsIndices, 'Label'] == 0])
        ExpensiveHList.append(HNumber)
        ExpensiveLList.append(LNumber)
        
        # Retrain the model and generate cheap points
        NsIndices = np.concatenate((InitialIndices, NsIndices), axis = None)
        print(len(NsIndices))
        featureTrain = permuData.loc[NsIndices, 'X1':'X12'].values
        labelTrain = permuData.loc[NsIndices, 'Label'].values
        model.fit(featureTrain, labelTrain)
        ExcludedIndices = np.setdiff1d(np.array(permuData.index), NsIndices)
        CheapPoint = model.predict(permuData.loc[ExcludedIndices, 'X1':'X12'])
        Prob = model.predict_proba(permuData.loc[ExcludedIndices, 'X1':'X12'])
        permuData.loc[ExcludedIndices, 'PredictLabel'] = CheapPoint
        permuData.loc[ExcludedIndices, 'Plow':'Phigh'] = Prob
        
        InitialIndices = NsIndices
        
        # Examine the metrics of the classifier for each iteration
        x = permuData.loc[NsIndices, 'X1':'X12']    
        y = permuData.loc[NsIndices, 'Label']
        
        kf = KFold(n_splits = 5)
        Confusion = pd.DataFrame(columns = ["TN", "FP", "FN", "TP"])
        
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
            Confusion = Confusion.append(pd.Series([CM[0,0],CM[0,1],CM[1,0],CM[1,1]], index = ["TN", "FP", "FN", "TP"]), ignore_index = True)
        
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
Average = np.mean(ListArray, axis = 0).reshape(80, 1)
Std = np.std(ListArray, axis = 0).reshape(80, 1)
Percentile90 = np.percentile(ListArray, 90, axis = 0).reshape(80,1)
Percentile10 = np.percentile(ListArray, 10, axis = 0).reshape(80,1)
Evaluations = (np.arange(80)*100) + 100
Evaluations = Evaluations.reshape(80, 1)
PrintingData = np.concatenate((Evaluations, Average, Std, Percentile10, Percentile90), axis = 1)
PrintingDataFrame = pd.DataFrame(PrintingData, columns = ['Evaluations', 'Mean', 'Std', 'Percentile10', 'Percentile90'])
#PrintingDataFrame.to_csv("/Users/dauku/Desktop/Python/MAE586/TwentyKnapsack/3-20/PrintingDataFrame.csv", index = False)

PrintingRecall = pd.DataFrame(np.concatenate((Evaluations, np.array(RecallList).reshape(80, 1)), axis = 1), columns = ["Evaluations", "Recall"])
#PrintingRecall.to_csv("/Users/dauku/Desktop/Python/MAE586/TwentyKnapsack/3-20/BernoulliProcessIndicators.csv", index = False)
