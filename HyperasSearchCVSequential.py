from hyperopt import Trials, STATUS_OK, STATUS_FAIL, tpe
from keras.layers import Input, Dense, Activation, Dropout
from keras.models import Model
from keras.regularizers import l1_l2
import keras
import keras.callbacks

from hyperas import optim
from hyperas.distributions import choice, uniform, conditional

import os
import math
import datetime
import numpy as np
import pickle
import argparse

import methodsMLinterns
import customAutoencoder

# Configure argument parser
parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-f','--features', help='Choose between a, z and a_z', required=True)
parser.add_argument('-e','--encoding', help='Dimension of the encoding', required=True)
parser.add_argument('-s','--superval', help='Percentage of training. The rest is used for super-validation', required=True)
parser.add_argument('-w','--weights-directory', help='Name of the directory where weights are saved', required=True)
args = vars(parser.parse_args())


features_1p4_extra_a_z =['aab', 'aac', 'aad', 'aae', 'aaf', 'aag', 'aah', 'abj', 'abm', 'abn', 'abo', 'abp', 'abq', 'abr', 'abs', 'abt', 'abu', 'abv', 'abw', 'abx', 'aby', 'abz', 'aca', 'acb', 'acc', 'acd', 'ace', 'acf', 'acr', 'acw', 'acx', 'acy', 'adi', 'adj', 'adl', 'ado', 'adp', 'adq', 'adr', 'ads', 'adt', 'adu', 'adv', 'adw', 'adx', 'ady', 'adz', 'aea', 'aeb', 'aec', 'aed', 'aee', 'aef', 'aeg', 'aeh', 'aei', 'aej', 'aek', 'ael', 'aem', 'aen', 'aeo', 'aep', 'aeq', 'aer', 'aes', 'aex', 'aey', 'aez', 'afa', 'afj', 'afl', 'afo', 'afp', 'afq', 'afr', 'afs', 'aft', 'afu', 'afv', 'afw', 'afx', 'afy', 'afz', 'aga', 'agb', 'agc', 'agd', 'age', 'agf', 'agg', 'agh', 'agi', 'agj', 'agk', 'agl', 'agm', 'agn', 'ago', 'agp', 'agq', 'agr', 'ags', 'agt', 'agu', 'agv', 'agw', 'agx', 'agy', 'ahf', 'ahg', 'ahh', 'ahi', 'ahj', 'ahk', 'ahl', 'ahm', 'ahn', 'aho', 'zhq', 'zhr', 'zhs', 'zht', 'zhu', 'zhv', 'zhw', 'ziy', 'zjb', 'zjc', 'zjd', 'zje', 'zjf', 'zjg', 'zjh', 'zji', 'zjj', 'zjk', 'zjl', 'zjm', 'zjn', 'zjo', 'zjp', 'zjq', 'zjr', 'zjs', 'zjt', 'zju', 'zkg', 'zkl', 'zkm', 'zkn', 'zkx', 'zky', 'zla', 'zld', 'zle', 'zlf', 'zlg', 'zlh', 'zli', 'zlj', 'zlk', 'zll', 'zlm', 'zln', 'zlo', 'zlp', 'zlq', 'zlr', 'zls', 'zlt', 'zlu', 'zlv', 'zlw', 'zlx', 'zly', 'zlz', 'zma', 'zmb', 'zmc', 'zmd', 'zme', 'zmf', 'zmg', 'zmh', 'zmm', 'zmn', 'zmo', 'zmp', 'zmy', 'zna', 'znd', 'zne', 'znf', 'zng', 'znh', 'zni', 'znj', 'znk', 'znl', 'znm', 'znn', 'zno', 'znp', 'znq', 'znr', 'zns', 'znt', 'znu', 'znv', 'znw', 'znx', 'zny', 'znz', 'zoa', 'zob', 'zoc', 'zod', 'zoe', 'zof', 'zog', 'zoh', 'zoi', 'zoj', 'zok', 'zol', 'zom', 'zon', 'zou', 'zov', 'zow', 'zox', 'zoy', 'zoz', 'zpa', 'zpb', 'zpc', 'zpd', 'zpe']
features_1p4_extra_a=[f for f in features_1p4_extra_a_z if f[0]=='a']
features_1p4_extra_z=[f for f in features_1p4_extra_a_z if f[0]=='z']

stocks = ['DNB', 'NRG', 'CL', 'ANTM', 'NEE', 'PAYX', 'VAR', 'NI', 'MNST', 'JNJ', 'TGNA', 'NOV', 'FIS', 'BLK', 'HBI', 'NVDA', 'DLTR', 'MRO', 'EMN', 'AMT', 'FLR', 'IBM', 'BK', 'NFX', 'AGN', 'LRCX', 'DIS', 'LH', 'C', 'MNK']
    
date_test_set = datetime.date(2016, 1, 1)
    
print("Loading Portfolio")
    
clf_portfolio_dic = methodsMLinterns.ClassificationPortfolio(stocks=stocks, minutes_forward=30)
clf_portfolio_dic.loadData()
clf_portfolio_dic.cleanUpData(features_1p4_extra_a_z)




def data():
    with open("pickles/CVhyperasSearchConfig.p",'rb') as f:
        x_train, y_train, sets, _, _, _, _ = pickle.load(f)
    return x_train, y_train, sets


def model(x_train, y_train, sets):
    if 'parameters' not in globals():
        global parameters
        parameters = []
    if 'val_loss1' not in globals():
        global val_loss1
        val_loss1 = []
    if 'val_loss2' not in globals():
        global val_loss2
        val_loss2 = []
    parameters.append(space)
    print(space)


    with open("pickles/CVhyperasSearchConfig.p",'rb') as f:
        _, _, _, feat, encod_dim, architecture, weightsDir = pickle.load(f)
    print(feat, encod_dim, architecture)
    
    autoencoderLoss = 'cosine_proximity'
    d = {{choice(numpy.arange(10,81,5)/100)}}
    l2={{choice([.0025, .005, .01, .02, .04, .08])}}
    inputNoise = 0
    batch_size = 256
    modelType = customAutoencoder.ModelType.Sequential


    log_name = datetime.datetime.today().strftime('%Y-%m-%d_%H:%M:%S')
    
    models = []
    freezed_models = []
    for train, val in sets:
        merged_model = customAutoencoder.Autoencoder(architecture, modelType, weightsDirectory=weightsDir, dropout=d, inputNoise=inputNoise, l1reg=0, l2reg=l2, autoencoderLoss=autoencoderLoss)
        merged_model.buildAutoencoder()
        models.append(merged_model)
    
    print("\n\nTraining First Model\n\n")
    val_logit_acc = 0
    for i, m in enumerate(models):
        print("\n\tFold %s\n"%i)
        checkpointName0="%s/test_%s_%s_%s_f0_%s.hdf5" % (weightsDir, log_name, feat, encod_dim, i)
        train_idx = sets[i][0]
        val_idx = sets[i][1]
        
        m.fit(x_train[train_idx,:], y_train[train_idx,:], x_train[val_idx,:], y_train[val_idx,:], epochs=200, batch=batch_size, checkpointName=checkpointName0)
            
        vl, _ = m.model1.evaluate(x_train[val_idx,:], x_train[val_idx,:], verbose=0)
        val_loss1.append(vl)
        if math.isnan(vl):
            val_loss2.append(vl)
            with open("%s/CVhyperasSearchParameters_%s_%s.p" % (weightsDir, feat, encod_dim), 'wb') as f:
                pickle.dump( [parameters,val_loss1,val_loss2], f, protocol=pickle.HIGHEST_PROTOCOL)
                return {'loss': 0, 'status': STATUS_FAIL, 'model': None}
 
        m.freezeAutoencoder(freeze=True)
        m.loadFromWeights(checkpointName0)

        freezed_models.append(m)
          
        _, vlacc = m.model1.evaluate(x_train[val_idx,:], x_train[val_idx,:], verbose=0)
        val_logit_acc += vlacc
        print("VLMAE ",vlacc)



    print("\n\nTraining Freezed Model\n\n")
    val_logit_acc_freezed = 0
    for i, m in enumerate(freezed_models):
        print("\n\tFold %s\n"%i)
        checkpointName1="%s/test_%s_%s_%s_f1_%s.hdf5" % (weightsDir, log_name, feat, encod_dim, i)
        train_idx = sets[i][0]
        val_idx = sets[i][1]
        
        m.fit(x_train[train_idx,:], y_train[train_idx,:], x_train[val_idx,:], y_train[val_idx,:], epochs=100, batch=batch_size, checkpointName=checkpointName1)
        
        vl, _ = m.model.evaluate(x_train[val_idx,:], y_train[val_idx,:], verbose=0)
        val_loss2.append(vl)
        with open("%s/CVhyperasSearchParameters_%s_%s.p" % (weightsDir, feat, encod_dim), 'wb') as f:
            pickle.dump( [parameters,val_loss1,val_loss2], f, protocol=pickle.HIGHEST_PROTOCOL)
        if math.isnan(vl):
            return {'loss': 0, 'status': STATUS_FAIL, 'model': None}

        m.loadFromWeights(checkpointName1)

        _, vlacc = m.model.evaluate(x_train[val_idx,:], y_train[val_idx,:] , verbose=0)
        val_logit_acc_freezed += vlacc
        print("VLACC ",vlacc)
    
    return {'loss': -val_logit_acc_freezed, 'status': STATUS_OK, 'model': merged_model}




def hyperasSearchCV(features, encoding_dim, evals, trainPercentage=100, weightsDir="weightsCV", useValDate=False, useShuffledDays=False):
    if not os.path.isdir(weightsDir):
        os.makedirs(weightsDir)
    
    if not useValDate and not useShuffledDays:
        clf_portfolio_dic.getTrainTestSetDate(date_test_set)
    elif useValDate and trainPercentage == 80:
        clf_portfolio_dic.getTrainValTestSetDate(datetime.date(2015, 5, 1), date_test_set)
    elif useValDate and trainPercentage == 90:
        clf_portfolio_dic.getTrainValTestSetDate(datetime.date(2015, 8, 24), date_test_set)
    elif useShuffledDays and not useValDate:
        clf_portfolio_dic.getTrainValTestShuffledDaysSetDate(date_test_set, percentageTrain=trainPercentage)



    architecture = [len(features), 100, encoding_dim]
    if features == features_1p4_extra_a:
        feat = "a"
    elif features == features_1p4_extra_z:
        feat = "z"
    elif features == features_1p4_extra_a_z:
        feat = "a_z"

    print("Features %s and encoding dimension %s"%(feat,encoding_dim))

    def prepareData(features):
        x_train = np.array([], dtype=np.float32).reshape(0,len(features))
        y_train = np.array([], dtype=np.float32).reshape(0,1)
    
        for k, stock in enumerate(clf_portfolio_dic.stocks):
            name = stock + str(clf_portfolio_dic.minutes_forward)
            if k==0:
                x_train, y_train = clf_portfolio_dic.X_train_dic[name][features].as_matrix(),(clf_portfolio_dic.y_train_dic[name]+1)/2
            else:
                x_train = np.concatenate((x_train,clf_portfolio_dic.X_train_dic[name][features].as_matrix()),axis=0)
                y_train = np.concatenate((y_train,(clf_portfolio_dic.y_train_dic[name]+1)/2),axis=0)

        y_t = np.zeros((y_train.shape[0], 2))
        y_t[np.arange(y_train.shape[0]), y_train.astype('int32')] = 1
        y_train = y_t
    
        return x_train,y_train



    print("Loading Data")
    assert(trainPercentage in [80,90,100])
    with open("pickles/CV_setsIndexes_%sp.p"%trainPercentage,'rb') as f:
        sets = pickle.load(f)
   
    X_train, y_train = prepareData(features)

    with open("pickles/CVhyperasSearchConfig.p",'wb') as f:
        pickle.dump( [X_train, y_train, sets, feat, encoding_dim, architecture, weightsDir], f, protocol=pickle.HIGHEST_PROTOCOL)

    best_run, _ = optim.minimize(model=model,
                                 data=data,
                                 algo=tpe.suggest,
                                 max_evals=evals,
                                 trials=Trials())

    with open("%s/CVhyperasSearchBestParameters_%s_%s.p" % (weightsDir, feat, encoding_dim),'wb') as f:
        pickle.dump( best_run, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open("%s/CVhyperasSearchParameters_%s_%s.p" % (weightsDir, feat, encoding_dim),'rb') as f:
        parameters,val_loss1,val_loss2 = pickle.load(f)
    print("----------")
    for p in parameters:
        print(p)
    print("----------")
    for v in val_loss1:
        print(v)
    print("----------")
    for v in val_loss2:
        print(v)



if __name__ == "__main__":
    '''
    Check parameters
    '''
    assert("weightsCV" in args["weights_directory"])
    assert(int(args["superval"]) in [80, 90, 100])
    assert(int(args["encoding"]) in [30, 50, 70])
    assert(args["features"] in ["a","z","a_z"])
    if args["features"] == "a":
        ft = features_1p4_extra_a
    elif args["features"] == "z":
        ft = features_1p4_extra_z
    elif args["features"] == "a_z":
        ft = features_1p4_extra_a_z
    '''
    Launch function
    '''
    hyperasSearchCV(features = ft, encoding_dim=int(args["encoding"]), evals=40, trainPercentage=int(args["superval"]), weightsDir=args["weights_directory"], useShuffledDays=True)

