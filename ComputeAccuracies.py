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
    X_test = np.array([], dtype=np.float32).reshape(0,len(features))
    y_test = np.array([], dtype=np.float32).reshape(0,1)
    X_val = np.array([], dtype=np.float32).reshape(0,len(features))
    y_val = np.array([], dtype=np.float32).reshape(0,1)
    ret_test = np.array([], dtype=np.float32).reshape(0,1)
    
    # Concatenate the stocks
    for k, stock in enumerate(clf_portfolio_dic.stocks):
        name = stock + str(clf_portfolio_dic.minutes_forward)
        if k==0:
            X_test, y_test = clf_portfolio_dic.X_test_dic[name][features].as_matrix(), (clf_portfolio_dic.y_test_dic[name]+1)/2
            X_val, y_val = clf_portfolio_dic.X_val_dic[name][features].as_matrix(), (clf_portfolio_dic.y_val_dic[name]+1)/2
            ret_test = clf_portfolio_dic.return_test_dic[name]
        else:
            X_test = np.concatenate((X_test,clf_portfolio_dic.X_test_dic[name][features].as_matrix()),axis=0)
            y_test = np.concatenate((y_test,(clf_portfolio_dic.y_test_dic[name]+1)/2),axis=0)
            X_val = np.concatenate((X_val,clf_portfolio_dic.X_val_dic[name][features].as_matrix()),axis=0)
            y_val = np.concatenate((y_val,(clf_portfolio_dic.y_val_dic[name]+1)/2),axis=0)
            ret_test = np.concatenate((ret_test,clf_portfolio_dic.return_test_dic[name]),axis=0)
    
    # Transform Ys to one-hot vectors
    y_t = np.zeros((y_test.shape[0], 2))
    y_t[np.arange(y_test.shape[0]), y_test.astype('int32')] = 1
    y_test = y_t

    y_t = np.zeros((y_val.shape[0], 2))
    y_t[np.arange(y_val.shape[0]), y_val.astype('int32')] = 1
    y_val = y_t

    return X_val, y_val, X_test, y_test
# Define metrics
def accuracy(y_true, y_pred):
    return (100 * (np.argmax(y_pred,1) == np.argmax(y_true,1))).mean()


### Test on stocks
def testOnStocks(filenames):
    architecture = [len(feat),100,encoding_dim]
    x_superval, y_superval, x_test, y_test = prepareData(feat)
    
    autoList = []
    for name in filenames:
        print(name)
        ensemble = AutoencoderEnsemble(name)
        ensemble.loadModels(ModelType.Merged, architecture, weightsDir)
        autoList.append(ensemble)

    
    n = len(autoList)
    acc_test_autoencoder = np.zeros((n,))
    acc_test_autoencoder2 = np.zeros((n,))
    acc_test_autoencoder3 = np.zeros((n,))
    acc_superval_autoencoder = np.zeros((n,))

    # compute accuracies
    for i, a in enumerate(autoList):
        print(i)
        hardVoting, softVoting, _, unanimityVoting,_ = a.predict(x_test)
        y_pred_test = softVoting
        y_pred_test2 = unanimityVoting
        y_pred_test3 = hardVoting
        _, _, _, y_pred_superval, _ = a.predict(x_superval)

        acc_test_autoencoder[i] = accuracy(y_test, y_pred_test)
        mask = np.sum(y_pred_test2,axis=1)>0
        acc_test_autoencoder2[i] = accuracy(y_test[mask], y_pred_test2[mask])
        acc_test_autoencoder3[i] = accuracy(y_test, y_pred_test3)
        acc_superval_autoencoder[i] = accuracy(y_superval, y_pred_superval)

    return(acc_test_autoencoder, acc_test_autoencoder2,
           acc_test_autoencoder3, acc_superval_autoencoder)


# Compute the accuracies
acc_test_vec, acc_test_vec2, acc_test_vec3, acc_superval_vec  = testOnStocks(flnames)

# Save the results
results = []
for i,n in enumerate(flnames):
    results.append([acc_test_vec[i],
                    acc_test_vec2[i],
                    acc_test_vec3[i],
                    acc_superval_vec[i]])
                   
with open("_pickles/results_%s_%s.p"%(int(sys.argv[1]),int(sys.argv[2])),'wb') as f:
    pickle.dump( results, f, protocol=pickle.HIGHEST_PROTOCOL)

print("\nDone")
