import math
import datetime
import numpy as np
import pickle

import methodsMLinterns
import customAutoencoder

stocks = ['DNB', 'NRG', 'CL', 'ANTM', 'NEE', 'PAYX', 'VAR', 'NI', 'MNST', 'JNJ', 'TGNA', 'NOV', 'FIS', 'BLK', 'HBI', 'NVDA', 'DLTR', 'MRO', 'EMN', 'AMT', 'FLR', 'IBM', 'BK', 'NFX', 'AGN', 'LRCX', 'DIS', 'LH', 'C', 'MNK']


features_1p4_extra_a_z =['aab', 'aac', 'aad', 'aae', 'aaf', 'aag', 'aah', 'abj', 'abm', 'abn', 'abo', 'abp', 'abq', 'abr', 'abs', 'abt', 'abu', 'abv', 'abw', 'abx', 'aby', 'abz', 'aca', 'acb', 'acc', 'acd', 'ace', 'acf', 'acr', 'acw', 'acx', 'acy', 'adi', 'adj', 'adl', 'ado', 'adp', 'adq', 'adr', 'ads', 'adt', 'adu', 'adv', 'adw', 'adx', 'ady', 'adz', 'aea', 'aeb', 'aec', 'aed', 'aee', 'aef', 'aeg', 'aeh', 'aei', 'aej', 'aek', 'ael', 'aem', 'aen', 'aeo', 'aep', 'aeq', 'aer', 'aes', 'aex', 'aey', 'aez', 'afa', 'afj', 'afl', 'afo', 'afp', 'afq', 'afr', 'afs', 'aft', 'afu', 'afv', 'afw', 'afx', 'afy', 'afz', 'aga', 'agb', 'agc', 'agd', 'age', 'agf', 'agg', 'agh', 'agi', 'agj', 'agk', 'agl', 'agm', 'agn', 'ago', 'agp', 'agq', 'agr', 'ags', 'agt', 'agu', 'agv', 'agw', 'agx', 'agy', 'ahf', 'ahg', 'ahh', 'ahi', 'ahj', 'ahk', 'ahl', 'ahm', 'ahn', 'aho', 'zhq', 'zhr', 'zhs', 'zht', 'zhu', 'zhv', 'zhw', 'ziy', 'zjb', 'zjc', 'zjd', 'zje', 'zjf', 'zjg', 'zjh', 'zji', 'zjj', 'zjk', 'zjl', 'zjm', 'zjn', 'zjo', 'zjp', 'zjq', 'zjr', 'zjs', 'zjt', 'zju', 'zkg', 'zkl', 'zkm', 'zkn', 'zkx', 'zky', 'zla', 'zld', 'zle', 'zlf', 'zlg', 'zlh', 'zli', 'zlj', 'zlk', 'zll', 'zlm', 'zln', 'zlo', 'zlp', 'zlq', 'zlr', 'zls', 'zlt', 'zlu', 'zlv', 'zlw', 'zlx', 'zly', 'zlz', 'zma', 'zmb', 'zmc', 'zmd', 'zme', 'zmf', 'zmg', 'zmh', 'zmm', 'zmn', 'zmo', 'zmp', 'zmy', 'zna', 'znd', 'zne', 'znf', 'zng', 'znh', 'zni', 'znj', 'znk', 'znl', 'znm', 'znn', 'zno', 'znp', 'znq', 'znr', 'zns', 'znt', 'znu', 'znv', 'znw', 'znx', 'zny', 'znz', 'zoa', 'zob', 'zoc', 'zod', 'zoe', 'zof', 'zog', 'zoh', 'zoi', 'zoj', 'zok', 'zol', 'zom', 'zon', 'zou', 'zov', 'zow', 'zox', 'zoy', 'zoz', 'zpa', 'zpb', 'zpc', 'zpd', 'zpe']
features_1p4_extra_a=[f for f in features_1p4_extra_a_z if f[0]=='a']
features_1p4_extra_z=[f for f in features_1p4_extra_a_z if f[0]=='z']

print("Loading Portfolio")

date_test_set = datetime.date(2016, 1, 1)

clf_portfolio_dic = methodsMLinterns.ClassificationPortfolio(stocks=stocks, minutes_forward=30)
clf_portfolio_dic.loadData()
clf_portfolio_dic.cleanUpData(features_1p4_extra_a_z)
clf_portfolio_dic.getTrainTestSetDate(date_test_set)


### Group together all the stocks

def prepareData(features):
    X_train = np.array([], dtype=np.float32).reshape(0,len(features))
    y_train = np.array([], dtype=np.float32).reshape(0,1)
    X_test = np.array([], dtype=np.float32).reshape(0,len(features))
    y_test = np.array([], dtype=np.float32).reshape(0,1)
    
    for k, stock in enumerate(clf_portfolio_dic.stocks):
        name = stock + str(clf_portfolio_dic.minutes_forward)
        if k==0:
            X_train, y_train = clf_portfolio_dic.X_train_dic[name][features].as_matrix(),(clf_portfolio_dic.y_train_dic[name]+1)/2
            X_test, y_test = clf_portfolio_dic.X_test_dic[name][features].as_matrix(), (clf_portfolio_dic.y_test_dic[name]+1)/2
        else:
            X_train = np.concatenate((X_train,clf_portfolio_dic.X_train_dic[name][features].as_matrix()),axis=0)
            y_train = np.concatenate((y_train,(clf_portfolio_dic.y_train_dic[name]+1)/2),axis=0)
            X_test = np.concatenate((X_test,clf_portfolio_dic.X_test_dic[name][features].as_matrix()),axis=0)
            y_test = np.concatenate((y_test,(clf_portfolio_dic.y_test_dic[name]+1)/2),axis=0)

    # Transform to one hot vectors
    y_t = np.zeros((y_train.shape[0], 2))
    y_t[np.arange(y_train.shape[0]), y_train.astype('int32')] = 1
    y_train = y_t
    
    y_t = np.zeros((y_test.shape[0], 2))
    y_t[np.arange(y_test.shape[0]), y_test.astype('int32')] = 1
    y_test = y_t
    
    return X_train,y_train,X_test,y_test


## Create the vectors for Grid Search

ds=[.1, .3, .5, .7]
l2s=[.005, .02, .08]
batch_sizes=[128, 256, 512]
ins=[0., .05, .1]
autoencLosses=['cosine_proximity','mean_squared_error']


print("Creating datasets")

## Create Train and Validation sets (60% and 40%)

X_train_a, y_train_a, X_test_a, y_test_a = prepareData(features_1p4_extra_a)
X_train_z, y_train_z, X_test_z, y_test_z = prepareData(features_1p4_extra_z)
X_train_a_z, y_train_a_z, X_test_a_z, y_test_a_z = prepareData(features_1p4_extra_a_z)

n = X_train_a.shape[0]
valSize = int(0.4*n)

validation_idx = np.random.randint(X_train_a.shape[0], size=valSize)
mask = np.array([(i in validation_idx) for i in range(n)])

X_train_a, y_train_a, X_val_a, y_val_a = X_train_a[~mask,:], y_train_a[~mask,:], X_train_a[mask,:], y_train_a[mask,:]
X_train_z, y_train_z, X_val_z, y_val_z = X_train_z[~mask,:], y_train_z[~mask,:], X_train_z[mask,:], y_train_z[mask,:]
X_train_a_z, y_train_a_z, X_val_a_z, y_val_a_z = X_train_a_z[~mask,:], y_train_a_z[~mask,:], X_train_a_z[mask,:], y_train_a_z[mask,:]

## Iterate over the different models

def gridIndexes(numRuns):
    multiplier = 10
    ds_idx = np.random.choice(len(ds), 2*numRuns)
    l2s_idx = np.random.choice(len(l2s), 2*numRuns)
    batch_sizes_idx = np.random.choice(len(batch_sizes), 2*numRuns)
    ins_idx = np.random.choice(len(ins), 2*numRuns)
    autoencLosses_idx = np.random.choice(len(autoencLosses), 2*numRuns)
    
    # check uniqueness
    stacked = np.vstack([ds_idx,l2s_idx,batch_sizes_idx,ins_idx,autoencLosses_idx])
    unique = np.unique(stacked, axis=1)
    assert(unique.shape[1]>=numRuns)
    
    chosenParameters = [np.asarray(ds)[(unique[0,:numRuns])],
                        np.asarray(l2s)[(unique[1,:numRuns])],
                        np.asarray(batch_sizes)[(unique[2,:numRuns])],
                        np.asarray(ins)[(unique[3,:numRuns])],
                        np.asarray(autoencLosses)[(unique[4,:numRuns])]]
        
    return zip(chosenParameters[0], chosenParameters[1], chosenParameters[2],
               chosenParameters[3], chosenParameters[4], ) ,chosenParameters

print("Starting training")

numberOfGridRuns = 100
listOfParameters = {}
for k, features in enumerate([features_1p4_extra_a_z]):  # features_1p4_extra_a, features_1p4_extra_z, 
    for j,encoding_size in enumerate([30]): # 50, 70
        architecture = [len(features),100,encoding_size]
        zippedIndexes, indexesArray = gridIndexes(numberOfGridRuns)
        # create name of entry in dictionary of parameters (a bit wordy...)
        if features == features_1p4_extra_a:
            feat = "a"
        elif features == features_1p4_extra_z:
            feat = "z"
        elif features == features_1p4_extra_a_z:
            feat = "a_z"
        listOfParameters["%s_%s"%(feat,encoding_size)] = indexesArray
        for d, l2, batchSize, inputNoise, loss in zippedIndexes:
            print("\n\n\t (d%s,l%s,batch%s,in%s,%s) First part:\n"%(d,l2,batchSize, inputNoise, loss))
            auto = customAutoencoder.Autoencoder(architecture,dropout=d,inputNoise=inputNoise,
                                                 l2reg=l2,autoencoderLoss=loss)
            auto.buildAutoencoder()
                                                 
            if features == features_1p4_extra_a:
                X_train,y_train,X_val,y_val = X_train_a, y_train_a, X_val_a, y_val_a
            elif features == features_1p4_extra_z:
                X_train,y_train,X_val,y_val = X_train_z, y_train_z, X_val_z, y_val_z
            elif features == features_1p4_extra_a_z:
                X_train,y_train,X_val,y_val = X_train_a_z, y_train_a_z, X_val_a_z, y_val_a_z
                    
            auto.fit(X_train,y_train,X_val,y_val,batch=batchSize,epochs=200)
                                                                         
            chkpt_0 = "weights/%s.hdf5"%(auto.name)
                                                                             
            auto.freezeAutoencoder(True)
                                                                                 
            print("\n\n\t (d%s,l%s,batch%s,in%s,%s) Second part:\n"%(d,l2,batchSize, inputNoise, loss))
                                                                                     
            auto.loadFromWeights(chkpt_0)
            auto.fit(X_train,y_train,X_val,y_val,batch=batchSize,epochs=100)

print("Saving parameters used and validation set to listOfParameters_RandGrid.p")

with open("pickles/listOfParameters_RandGrid.p",'wb') as f:
    pickle.dump( [listOfParameters, validation_idx], f, protocol=pickle.HIGHEST_PROTOCOL)

print("Finished properly")
