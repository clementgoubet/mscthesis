import sys
import time
import os

import math
import datetime
import numpy as np
import pickle
import glob
from sklearn.model_selection import train_test_split

import methodsMLinterns
from customAutoencoder import ModelType, AutoencoderEnsemble, ExperimentPerformance, Autoencoder

assert(len(sys.argv)==3)

features_1p4_extra =['aab', 'aac', 'aad', 'aae', 'aaf', 'aag', 'aah', 'abj', 'abm', 'abn', 'abo', 'abp', 'abq', 'abr', 'abs', 'abt', 'abu', 'abv', 'abw', 'abx', 'aby', 'abz', 'aca', 'acb', 'acc', 'acd', 'ace', 'acf', 'acr', 'acw', 'acx', 'acy', 'adi', 'adj', 'adl', 'ado', 'adp', 'adq', 'adr', 'ads', 'adt', 'adu', 'adv', 'adw', 'adx', 'ady', 'adz', 'aea', 'aeb', 'aec', 'aed', 'aee', 'aef', 'aeg', 'aeh', 'aei', 'aej', 'aek', 'ael', 'aem', 'aen', 'aeo', 'aep', 'aeq', 'aer', 'aes', 'aex', 'aey', 'aez', 'afa', 'afj', 'afl', 'afo', 'afp', 'afq', 'afr', 'afs', 'aft', 'afu', 'afv', 'afw', 'afx', 'afy', 'afz', 'aga', 'agb', 'agc', 'agd', 'age', 'agf', 'agg', 'agh', 'agi', 'agj', 'agk', 'agl', 'agm', 'agn', 'ago', 'agp', 'agq', 'agr', 'ags', 'agt', 'agu', 'agv', 'agw', 'agx', 'agy', 'ahf', 'ahg', 'ahh', 'ahi', 'ahj', 'ahk', 'ahl', 'ahm', 'ahn', 'aho', 'zhq', 'zhr', 'zhs', 'zht', 'zhu', 'zhv', 'zhw', 'ziy', 'zjb', 'zjc', 'zjd', 'zje', 'zjf', 'zjg', 'zjh', 'zji', 'zjj', 'zjk', 'zjl', 'zjm', 'zjn', 'zjo', 'zjp', 'zjq', 'zjr', 'zjs', 'zjt', 'zju', 'zkg', 'zkl', 'zkm', 'zkn', 'zkx', 'zky', 'zla', 'zld', 'zle', 'zlf', 'zlg', 'zlh', 'zli', 'zlj', 'zlk', 'zll', 'zlm', 'zln', 'zlo', 'zlp', 'zlq', 'zlr', 'zls', 'zlt', 'zlu', 'zlv', 'zlw', 'zlx', 'zly', 'zlz', 'zma', 'zmb', 'zmc', 'zmd', 'zme', 'zmf', 'zmg', 'zmh', 'zmm', 'zmn', 'zmo', 'zmp', 'zmy', 'zna', 'znd', 'zne', 'znf', 'zng', 'znh', 'zni', 'znj', 'znk', 'znl', 'znm', 'znn', 'zno', 'znp', 'znq', 'znr', 'zns', 'znt', 'znu', 'znv', 'znw', 'znx', 'zny', 'znz', 'zoa', 'zob', 'zoc', 'zod', 'zoe', 'zof', 'zog', 'zoh', 'zoi', 'zoj', 'zok', 'zol', 'zom', 'zon', 'zou', 'zov', 'zow', 'zox', 'zoy', 'zoz', 'zpa', 'zpb', 'zpc', 'zpd', 'zpe']
features_1p4_extra_a=[f for f in features_1p4_extra if f[0]=='a']
features_1p4_extra_z=[f for f in features_1p4_extra if f[0]=='z']

#########################
# Modifiable Parameters #
#########################
feat = features_1p4_extra_z
encoding_dim = 30
weightsDir = "weightsCV_run56"
trainPercentage = 90
#########################

stocks = ['DNB', 'NRG', 'CL', 'ANTM', 'NEE', 'PAYX', 'VAR', 'NI', 'MNST', 'JNJ', 'TGNA', 'NOV', 'FIS', 'BLK', 'HBI', 'NVDA', 'DLTR', 'MRO', 'EMN', 'AMT', 'FLR', 'IBM', 'BK', 'NFX', 'AGN', 'LRCX', 'DIS', 'LH', 'C', 'MNK']


# Extract the names of the saved models in weightsDir and group folds together
filenames = []
unique = []
i=0
temp=[]
for file in glob.glob("%s/test_*_%s*.hdf5"%(weightsDir,encoding_dim)):
    name = file[len(weightsDir)+6:-7]
    if name not in unique:
        unique.append(name)
for u in unique:
    filenames.append([f[len(weightsDir)+1:-5] for f in glob.glob("%s/test_*_%s*.hdf5"%(weightsDir,encoding_dim)) if u in f])
    print(filenames[-1])

print(len(filenames))

# Define the subset of models to analyse (thanks to command line arguments)
flnames = filenames[int(sys.argv[1]):int(sys.argv[2])]

# Load Portfolio Data
date_test_set = datetime.date(2016, 5, 1)
clf_portfolio_dic = methodsMLinterns.ClassificationPortfolio(stocks=stocks, minutes_forward=30)
#clf_portfolio_dic.loadDataSingleFile()
clf_portfolio_dic.loadData()
clf_portfolio_dic.cleanUpData(features_1p4_extra)
clf_portfolio_dic.getTrainValTestShuffledDaysSetDate(date_test_set, percentageTrain=trainPercentage)

def prepareData(features):
    # Create empty arrays
    X_train = np.array([], dtype=np.float32).reshape(0,len(features))
    y_train = np.array([], dtype=np.float32).reshape(0,1)
    X_test = np.array([], dtype=np.float32).reshape(0,len(features))
    y_test = np.array([], dtype=np.float32).reshape(0,1)
    X_val = np.array([], dtype=np.float32).reshape(0,len(features))
    y_val = np.array([], dtype=np.float32).reshape(0,1)
    
    # Concatenate the stocks
    for k, stock in enumerate(clf_portfolio_dic.stocks):
        name = stock + str(clf_portfolio_dic.minutes_forward)
        if k==0:
            X_test, y_test = clf_portfolio_dic.X_test_dic[name][features].as_matrix(), (clf_portfolio_dic.y_test_dic[name]+1)/2
            X_val, y_val = clf_portfolio_dic.X_val_dic[name][features].as_matrix(), (clf_portfolio_dic.y_val_dic[name]+1)/2
        else:
            X_test = np.concatenate((X_test,clf_portfolio_dic.X_test_dic[name][features].as_matrix()),axis=0)
            y_test = np.concatenate((y_test,(clf_portfolio_dic.y_test_dic[name]+1)/2),axis=0)
            X_val = np.concatenate((X_val,clf_portfolio_dic.X_val_dic[name][features].as_matrix()),axis=0)
            y_val = np.concatenate((y_val,(clf_portfolio_dic.y_val_dic[name]+1)/2),axis=0)
    X_train = clf_portfolio_dic.trainSet.drop(["stock","y","uniqueDate","ret"],axis=1)[features].as_matrix()
    y_train = (clf_portfolio_dic.trainSet["y"].as_matrix()+1)/2


    
    # Transform to one hot vectors
    y_t = np.zeros((y_train.shape[0], 2))
    y_t[np.arange(y_train.shape[0]), y_train.astype('int32')] = 1
    y_train = y_t
    
    y_t = np.zeros((y_test.shape[0], 2))
    y_t[np.arange(y_test.shape[0]), y_test.astype('int32')] = 1
    y_test = y_t
    
    y_t = np.zeros((y_val.shape[0], 2))
    y_t[np.arange(y_val.shape[0]), y_val.astype('int32')] = 1
    y_val = y_t
    
    print(X_train.shape,y_train.shape)
    print(X_val.shape,y_val.shape)
    return X_train, y_train, X_val, y_val, X_test, y_test, clf_portfolio_dic.sets



# Define metrics
def accuracy(y_true, y_pred):
    return (100 * (np.argmax(y_pred,1) == np.argmax(y_true,1))).mean()

def precision(y_true, y_pred):
    return( ((np.argmax(y_pred,1) == 1) & (np.argmax(y_true,1) == 1)).sum()
                / (np.argmax(y_pred,1) == 1).sum() )

def recall(y_true, y_pred):
    return( ((np.argmax(y_pred,1) == 1) & (np.argmax(y_true,1) == 1)).sum()
           / (((np.argmax(y_pred,1) == 1) & (np.argmax(y_true,1) == 1)).sum() +
              ((np.argmax(y_pred,1) == 0) & (np.argmax(y_true,1) == 1)).sum() ) )

def F1(y_true, y_pred):
    return 2 / ( 1/precision(y_true, y_pred) + 1/recall(y_true, y_pred) )

def balance(y):
    return (np.argmax(y,1) == 1).sum() / y.shape[0]



### Test on stocks separately

def testOnStocks(filenames):
    architecture = [len(feat),100,encoding_dim]
    
    X_train, Y_train, x_superval, y_superval, x_test, y_test, sets = prepareData(feat)
    
    autoList = []
    for name in filenames:
        temp = []
        for na in name:
            auto = Autoencoder(architecture, ModelType.Merged, weightsDir)
            auto.buildAutoencoder()
            auto.loadFromWeights("%s/%s.hdf5"%(weightsDir,na))
            temp.append(auto)
        autoList.append(temp)
    
    n = len(autoList)
    acc_train_autoencoder = np.zeros((n,3))
    acc_val_autoencoder = np.zeros((n,3))
    acc_superval_autoencoder = np.zeros((n,3))
    acc_test_autoencoder = np.zeros((n,3))

    for j in range(3):
        print("Fold ",j)
        # split train into val and train
        train_idx = sets[j][0]
        val_idx = sets[j][1]
        x_train, y_train = X_train[train_idx,:], Y_train[train_idx,:]
        x_val, y_val = X_train[val_idx,:], Y_train[val_idx,:]
        
        # compute accuracies
        for i, a in enumerate(autoList):
            print(i)
            # skip if no model saved
            if j >= len(a):
                continue
            y_pred_train = a[j].predict(x_train)
            y_pred_val = a[j].predict(x_val)
            y_pred_test = a[j].predict(x_test)
            y_pred_superval = a[j].predict(x_superval)

            acc_train_autoencoder[i,j] = accuracy(y_train, y_pred_train)
            acc_val_autoencoder[i,j] = accuracy(y_val, y_pred_val)
            acc_superval_autoencoder[i,j] = accuracy(y_superval, y_pred_superval)
            acc_test_autoencoder[i,j] = accuracy(y_test, y_pred_test)
            #print(balance(y_pred_train), balance(y_pred_val), balance(y_pred_superval), balance(y_pred_test))
            #print(balance(y_train), balance(y_val), balance(y_superval), balance(y_test))

    return(acc_train_autoencoder, acc_val_autoencoder, acc_test_autoencoder, acc_superval_autoencoder)


acc_train_vec, acc_val_vec, acc_test_vec, acc_superval_vec = testOnStocks(flnames)

results = []
for i,n in enumerate(flnames):
    results.append([ acc_train_vec[i].mean(),
                     acc_val_vec[i].mean(),
                     acc_test_vec[i].mean(),
                     acc_superval_vec[i].mean() ])
                   

with open("_pickles/results_avg_%s_%s.p"%(int(sys.argv[1]),int(sys.argv[2])),'wb') as f:
    pickle.dump( results, f, protocol=pickle.HIGHEST_PROTOCOL)

print("\nDone")
