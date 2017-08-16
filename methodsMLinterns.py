
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime 
import os
import random
import pickle
import dask.dataframe as dd

from math import sqrt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier

random_state = 0
home = os.path.expanduser('~')
directory= home + '/DataArrowClement/'


class ClassificationPortfolio(object):

    def __init__(self, stocks, minutes_forward):
        self.stocks = stocks
        self.N_stocks = len(stocks)
        self.minutes_forward = minutes_forward
        self.date_test_set = None
        self.year_test_set = None
        self.date_end = None
        self.X_dic = {}
        self.y_dic = {}
        self.X_clean_dic = {}
        self.y_clean_dic = {}
        self.X_train_dic = {}
        self.y_train_dic = {}
        self.X_val_dic = {}
        self.y_val_dic = {}
        self.X_test_dic = {}
        self.y_test_dic = {}
        self.y_pred_test_dic = {}
        self.anomaly_time_test_dic = {}
        self.daily_return_dic = {}
        self.return_dic = {}
        self.return_clean_dic = {}
        self.return_test_dic = {}
        self.return_train_dic = {}
        self.return_val_dic = {}


    def loadData(self):
        date_end = datetime.date(2010, 1, 1)
        with open("pickles/returnTrainTest_dic.p",'rb') as f:
            ret = pickle.load(f)
        for i, stock in enumerate(self.stocks):
            name = stock + str(self.minutes_forward)
            self.y_dic[name] = np.load(directory + 'y' + name +'.npy')
            re = ret[stock][~np.isnan(ret[stock])]
            self.return_dic[name] = re
            self.X_dic[name] = pd.read_csv(directory + 'X' + name +'.csv', 
                                           index_col=0, 
                                           parse_dates=True)
            date_end_ = self.X_dic[name].index[-1].date()
            if date_end_> date_end:
                date_end = date_end_
        self.date_end = date_end

    def loadDataSingleFile(self):
        date_end = datetime.date(2010, 1, 1)
        Xh5 = dd.read_hdf(directory+'X_normalized.h5', "us")
        X = pd.DataFrame(Xh5.compute())
        for i, stock in enumerate(self.stocks):
            name = stock + str(self.minutes_forward)
            self.y_dic[name] = X.loc[X["Stock"] == stock,"y"]
            self.return_dic[name] = X.loc[X["Stock"] == stock,"Ret"]
            self.X_dic[name] = X.loc[X["Stock"] == stock].drop(["Stock","y","Ret"],axis=1)
            date_end_ = self.X_dic[name].index[-1].date()
            if date_end_> date_end:
                date_end = date_end_
        self.date_end = date_end

    def cleanUpData(self, features):
        for i, stock in enumerate(self.stocks):
            name = stock + str(self.minutes_forward)
            print(name,self.X_dic[name][features].shape)
            indx_finite = pd.notnull(self.X_dic[name][features]).all(1).nonzero()[0]
            self.y_clean_dic[name] = np.array(self.y_dic[name])[indx_finite]
            self.return_clean_dic[name] = self.return_dic[name][indx_finite]
            self.X_clean_dic[name] = self.X_dic[name][features].dropna()
            

    def getTrainTestSetDate(self, date_init_test_set):
        self.date_init_test_set = date_init_test_set
        self.year_init_test = date_init_test_set.year
        for i, stock in enumerate(self.stocks):
            name = stock + str(self.minutes_forward)
            mask = (self.X_clean_dic[name].index.date >= date_init_test_set)
            n_test = mask.sum()
            self.y_train_dic[name] = self.y_clean_dic[name][:-n_test]
            self.y_test_dic[name] = self.y_clean_dic[name][-n_test:]
            self.X_train_dic[name] = self.X_clean_dic[name][:-n_test]
            self.X_test_dic[name] = self.X_clean_dic[name][-n_test:]
            self.return_test_dic[name] = self.return_clean_dic[name][-n_test:]
        self.year_end_test = self.X_test_dic[name].index[-1].year

    def getSecondOrderFeatures(self, features1_vec, features2_vec):
        for i, stock in enumerate(self.stocks):
            for f1, f2 in zip(features1_vec, features2_vec):
                name = stock + str(self.minutes_forward)
                self.X_train_dic[name][f1+f2] = self.X_train_dic[name][f1] * self.X_train_dic[name][f2]
                self.X_test_dic[name][f1+f2] = self.X_test_dic[name][f1] * self.X_test_dic[name][f2]
        features_second_order = []
        for f1, f2 in zip(features1_vec, features2_vec):   
            features_second_order.append(f1+f2)
        return features_second_order

    def getModelPCA(self, clf, features, n_components_vec):
        acc_train_mat = np.zeros((len(self.stocks),len(n_components_vec)))
        acc_test_mat = np.zeros((len(self.stocks),len(n_components_vec)))
        for k, stock in enumerate(self.stocks):
            if round(100*(1+k)/self.N_stocks, 2) % 20 == 0:
                print('done %s%%'%round(100*(1+k)/self.N_stocks))
            name = stock + str(self.minutes_forward)
            X_train, y_train = self.X_train_dic[name][features],self.y_train_dic[name]
            X_test, y_test = self.X_test_dic[name][features], self.y_test_dic[name]
            acc_train_vec = np.zeros(len(n_components_vec))
            acc_test_vec = np.zeros(len(n_components_vec))
            for j, n_components in enumerate(n_components_vec):
                try:
                    pca = PCA(n_components=n_components, random_state=random_state)
                    pca.fit(X_train)
                    X_train_pca = pca.transform(X_train)
                    X_test_pca = pca.transform(X_test)
                    clf.fit(X_train_pca, y_train)
                    y_pred_train = clf.predict(X_train_pca)
                    y_pred_test = clf.predict(X_test_pca)
                    acc_train = 100 * (y_pred_train == y_train).mean()
                    acc_test = 100 * (y_pred_test == y_test).mean()
                    acc_train_vec[j] = acc_train
                    acc_test_vec[j] = acc_test
                    print(name[:-2],
                      'n_components', n_components,
                      'accuracy train', round(acc_train, 2), 
                      'accuracy test', round(acc_test, 2))
                except np.linalg.linalg.LinAlgError as err:
                    pass
            fig, ax = plt.subplots(figsize=(8, 3))
            plt.ylim(49, 61)
            plt.xticks(n_components_vec, n_components_vec, color='red')
            plt.yticks(color='red')
            plt.plot(n_components_vec, acc_train_vec, 'bo--', ms=20)
            plt.plot(n_components_vec, acc_test_vec, 'go--', ms=20)

            plt.title(stock+' pca analysis', color='red')
            plt.show()
            acc_train_mat[k,:] = acc_train_vec
            acc_test_mat[k,:] = acc_test_vec
        return acc_train_mat,acc_test_mat



    def getModelSVD(self, clf, features, n_components_vec):
        acc_train_mat = np.zeros((len(self.stocks),len(n_components_vec)))
        acc_test_mat = np.zeros((len(self.stocks),len(n_components_vec)))
        for k, stock in enumerate(self.stocks):
            if round(100*(1+k)/self.N_stocks, 2) % 20 == 0:
                print('done %s%%'%round(100*(1+k)/self.N_stocks))
            name = stock + str(self.minutes_forward)
            X_train, y_train = self.X_train_dic[name][features],self.y_train_dic[name]
            X_test, y_test = self.X_test_dic[name][features], self.y_test_dic[name]
            acc_train_vec = np.zeros(len(n_components_vec))
            acc_test_vec = np.zeros(len(n_components_vec))
            for j, n_components in enumerate(n_components_vec):
                try:
                    svd = TruncatedSVD(n_components=n_components,
                                    algorithm='randomized', 
                                    n_iter=5, 
                                    random_state=None, 
                                    tol=0.0)

                    svd.fit(X_train, y=y_train)
                    X_train_svd = svd.fit_transform(X_train)
                    X_test_svd = svd.transform(X_test)
                    clf.fit(X_train_svd, y_train)
                    y_pred_train = clf.predict(X_train_svd)
                    y_pred_test = clf.predict(X_test_svd)
                    acc_train = 100 * (y_pred_train == y_train).mean()
                    acc_test = 100 * (y_pred_test == y_test).mean()
                    acc_train_vec[j] = acc_train
                    acc_test_vec[j] = acc_test
                    print(name[:-2],
                      'n_components', n_components,
                      'accuracy train', round(acc_train, 2), 
                      'accuracy test', round(acc_test, 2)
                      )
                except np.linalg.linalg.LinAlgError as err:
                    pass
            fig, ax = plt.subplots(figsize=(8, 3))
            plt.ylim(49, 61)
            plt.xticks(n_components_vec, n_components_vec, color='red')
            plt.yticks(color='red')
            plt.plot(n_components_vec, acc_train_vec, 'bo--', ms=20)
            plt.plot(n_components_vec, acc_test_vec, 'go--', ms=20)
            plt.title(stock+' svd analysis', color='red')
            plt.show()
            acc_train_mat[k,:] = acc_train_vec
            acc_test_mat[k,:] = acc_test_vec
        return acc_train_mat,acc_test_mat

    def getLogisticRegressionCV(self, features, Cs, cv):
        N_C = len(Cs)
        accs_cv = np.zeros(N_C)
        df_feature_importance = pd.DataFrame(0, index=self.stocks, columns=features)
        C_vec = np.zeros(self.N_stocks)
        acc_train_vec = np.zeros(self.N_stocks)
        acc_test_vec = np.zeros(self.N_stocks)
        for k, stock in enumerate(self.stocks):
            if round(100*(1+k)/self.N_stocks, 2) % 20 == 0:
                print('done %s%%'%round(100*(1+k)/self.N_stocks))
            name = stock + str(self.minutes_forward)
            X_train, y_train = self.X_train_dic[name][features],self.y_train_dic[name]
            X_test, y_test = self.X_test_dic[name][features], self.y_test_dic[name]
            clf = LogisticRegressionCV(Cs=Cs, penalty='l2', cv=cv, random_state=random_state, refit=True)
            clf.fit(X_train, y_train)
            y_pred_train = clf.predict(X_train)
            y_pred_test = clf.predict(X_test)
            df_feature_importance.ix[stock] = np.abs(clf.coef_[0]) / np.abs(clf.coef_[0]).sum()
            acc_train = 100 * (y_pred_train == y_train).mean()
            acc_test = 100 * (y_pred_test == y_test).mean()
            acc_train_vec[k] = acc_train
            acc_test_vec[k] = acc_test
            print(name[:-2], 
                  'accuracy train', round(acc_train, 2), 
                  'accuracy test', round(acc_test, 2))
            self.y_pred_test_dic[name] = y_pred_test
            self.anomaly_time_test_dic[name] = pd.to_datetime(X_test.index)
        x = np.arange(len(features))
        y = df_feature_importance.mean().values
        yerr = df_feature_importance.std().values
        fig, ax = plt.subplots(figsize=(12, 5))
        plt.xticks(x+0.5, features, color='red')
        plt.yticks(color='red')
        plt.bar(x-0.41, y, yerr=yerr, alpha=0.)
        plt.scatter(x, y, c=y, s=200, cmap=plt.get_cmap('Reds'))
        plt.tick_params(axis='both', which='major', labelsize=18)
        fig.autofmt_xdate()
        plt.show()
        return np.array(acc_train_vec), np.array(acc_test_vec), df_feature_importance

    def getModelRandomSearchCV(self, clf, cv, param_distributions, n_iter, features):
        acc_train_vec = np.zeros(self.N_stocks)
        acc_test_vec = np.zeros(self.N_stocks)
        for k, stock in enumerate(self.stocks):
            if round(100*(1+k)/self.N_stocks, 2) % 20 == 0:
                print('done %s%%'%round(100*(1+k)/self.N_stocks))
            name = stock + str(self.minutes_forward)
            X_train, y_train = self.X_train_dic[name][features],self.y_train_dic[name]
            X_test, y_test = self.X_test_dic[name][features], self.y_test_dic[name]
            grid_search = RandomizedSearchCV(estimator=clf, 
                                             param_distributions=param_distributions,
                                             cv=cv,
                                             n_iter=n_iter, 
                                             random_state=random_state,
                                             refit=True)
            grid_search.fit(X_train, y_train)
            best_clf = grid_search.best_estimator_
            print(best_clf)
            y_pred_train = best_clf.predict(X_train)
            y_pred_test = best_clf.predict(X_test)
            acc_train = 100 * np.mean(y_train == y_pred_train)
            acc_test =  100 * np.mean(y_test == y_pred_test)
            acc_train_vec[k] = acc_train
            acc_test_vec[k] = acc_test
            print(stock, 
                  'accuracy train', round(acc_train, 2), 
                  'accuracy test', round(acc_test, 2))
        return acc_train_vec, acc_test_vec

    def getModelEnsembleL1(self, Cs, features, cv, ratio_threshold):
        n_features_bounds = [(100, 95), (40, 30), (25, 20), (16, 14), (9, 6)]
        n_features = 0
        alphabet = ['a', 'b', 'c', 'd', 'e']
        N_models = len(alphabet)
        N_C = len(Cs)
        accs_cv = np.zeros(N_C)
        acc_train_vec = np.zeros(self.N_stocks)
        acc_test_vec = np.zeros(self.N_stocks)
        for k, stock in enumerate(self.stocks):
            np.random.seed = 0
            if round(100*(1+k)/self.N_stocks, 2) % 20 == 0:
                print('done %s%%'%round(100*(1+k)/self.N_stocks))
            name = stock + str(self.minutes_forward)
            X_train, y_train = self.X_train_dic[name][features], self.y_train_dic[name]
            X_test, y_test = self.X_test_dic[name][features], self.y_test_dic[name]
            N_train = len(y_train)
            N_test = len(y_test)
            anomaly_time_train = X_train.index
            anomaly_time_test = X_test.index
            features_ = {}
            C1 = 1
            model_dic = {}
            for n_features_bound, alpha in zip(n_features_bounds, alphabet):
                while (n_features > n_features_bound[0]) | (n_features < n_features_bound[1]):
                    model_l1 = LogisticRegression(C=C1, penalty='l1', random_state=random_state)
                    model_l1.fit(X_train, y_train)
                    coefs = model_l1.coef_
                    indx = np.argwhere(np.abs(coefs) > 1e-4)[:, 1]
                    n_features = len(indx)
                    if n_features >= n_features_bound[0]:
                        C1 /= 1.5
                    elif n_features <= n_features_bound[1]:
                        C1 *= 1.05
                    else : 
                        break
                columns_l1 = []
                for i in indx:
                    columns_l1.append(features[i])
                feature_name = name + str(alpha)
                features_[feature_name] = columns_l1
            for alpha in alphabet:
                feature_name = name + str(alpha)
                f = features_[feature_name]
                model_cv = LogisticRegressionCV(Cs=Cs, penalty='l2', cv=cv, random_state=random_state, refit=True)
                model_cv.fit(X_train[f], y_train)
                model_cv.fit(X_train[f], y_train)
                model_dic[feature_name] = model_cv

            ytrs = np.zeros(len(y_train))
            ytes = np.zeros(len(y_test))
            for alpha in alphabet:
                key = stock + str(self.minutes_forward) + alpha
                ytrs += model_dic[key].predict(X_train[features_[key]])
                ytes += model_dic[key].predict(X_test[features_[key]])

            for n_threshold in range(1, N_models+1):
                indx_train = np.argwhere(np.abs(ytrs) >= n_threshold)[:, 0]
                indx_test = np.argwhere(np.abs(ytes) >= n_threshold)[:, 0]
                if len(indx_train)/len(ytrs) < ratio_threshold:
                    n_threshold -= 1
                    indx_train = np.argwhere(np.abs(ytrs) >= n_threshold)[:, 0]
                    indx_test = np.argwhere(np.abs(ytes) >= n_threshold)[:, 0]
                    break
            yy_pred_train = np.sign(ytrs)
            yy_pred_test = np.sign(ytes)
            acc_train = 100 * np.mean(y_train[indx_train] == np.sign(ytrs[indx_train]))
            acc_test = 100 * np.mean(y_test[indx_test] == np.sign(ytes[indx_test]))

            ratio_train = round(100 * len(y_train[indx_train]) / N_train, 2)
            ratio_test = round(100 * len(y_test[indx_test]) / N_test, 2)
            print(stock,
                '   ratio', ratio_train, ratio_test, 
                '   accuracy train', acc_train, '   accuracy test',acc_test)
            anomaly_time_test = pd.to_datetime(anomaly_time_test)[indx_test]
            self.y_pred_test_dic[name] = np.sign(ytes[indx_test])
            self.anomaly_time_test_dic[name] = anomaly_time_test
        return np.array(acc_train_vec), np.array(acc_test_vec)

    def pltPearsonCorrelation(self, features, display=True):
        N_features = len(features)
        corr_sum = np.zeros((N_features, N_features))
        for k, stock in enumerate(self.stocks):
            name = stock + str(self.minutes_forward)
            corr_sum += self.X_train_dic[name][features].corr()
        if display:
            colormap = plt.cm.seismic
            fig, ax = plt.subplots(figsize=(22, 22))
            plt.title('PEARSON CORRELATION OF FEATURES', y=1.05, size=30, color='red')
            sns.heatmap(corr_sum/self.N_stocks,
                        linewidths=0.1,vmax=1.0, square=True, cmap=colormap, 
                        linecolor='white', annot=True)
            plt.yticks(0.5+np.arange(N_features), features[::-1], color='red')
            plt.yticks(rotation=0)
            plt.xticks(0.5+np.arange(N_features), features, color='red')
            plt.tick_params(axis='both', which='major', labelsize=20)
            fig.autofmt_xdate()
            plt.show()
        return corr_sum/self.N_stocks



########################
### ADDED BY CLEMENT ###
########################

## Create (Train + Val) Folds, super-validation set and test set.
## Extract test first with date then shuffle the rest by day
    def getTrainValTestShuffledDaysSetDate(self, date_init_test_set, percentageTrain):
        temp = {}
        for i, stock in enumerate(self.stocks):
            name = stock + str(self.minutes_forward)
            mask = (self.X_clean_dic[name].index.date >= date_init_test_set)
            n_test = mask.sum()
            temp[name] = self.X_clean_dic[name][:-n_test]
            self.y_test_dic[name] = self.y_clean_dic[name][-n_test:]
            self.X_test_dic[name] = self.X_clean_dic[name][-n_test:]
            self.return_test_dic[name] = self.return_clean_dic[name][-n_test:]
            temp[name].loc[:,"stock"] = name
            temp[name].loc[:,"y"] = self.y_clean_dic[name][:-n_test]
            temp[name].loc[:,"ret"] = self.return_clean_dic[name][:-n_test]
        # Concat all stocks for shuffle #
        trainFrames = [temp[stock + str(self.minutes_forward)] for stock in self.stocks]
        df = pd.concat(trainFrames)
        df["uniqueDate"] = df.index.normalize()
        groups = list(df.groupby("uniqueDate"))
        random.seed(random_state)
        random.shuffle(groups)
        CVindexes = np.zeros((3,),dtype='int32')
        counter=0
        trainGroup, valGroup = [], []
        for g, grp in groups:
            counter+=len(grp)
            # if past number of first fold, cut here
            if counter>=int(percentageTrain/100.*len(df)):
                valGroup.append(grp)
                if CVindexes[2]==0:
                    CVindexes[2] = counter-len(grp)
            else:
                trainGroup.append(grp)
                if counter>=int(2 * percentageTrain/100.*len(df)/3) and CVindexes[1]==0:
                    CVindexes[1] = counter
                elif counter>=int(percentageTrain/100.*len(df)/3) and CVindexes[0]==0:
                    CVindexes[0] = counter
        self.trainSet, self.valSet = pd.concat(trainGroup), pd.concat(valGroup)
        # create the 3 folds
        fold1 = np.arange(0,CVindexes[0])
        fold2 = np.arange(CVindexes[0], CVindexes[1])
        fold3 = np.arange(CVindexes[1], CVindexes[2])
        self.sets = [[np.hstack((fold1,fold2)), fold3], [np.hstack((fold2,fold3)), fold1], [np.hstack((fold1,fold3)), fold2]]
        # check for overlap
        for i in range(3):
            assert(len(list(set(self.sets[i][1]).intersection(self.sets[i][0]))) == 0)

        # Separate the stocks
        for stock in self.stocks:
            name = stock + str(self.minutes_forward)
            #self.X_train_dic[name] = self.trainSet.loc[self.trainSet['stock'] == name].drop(["stock","y","uniqueDate","ret"],axis=1)
            self.X_val_dic[name] = self.valSet.loc[self.valSet['stock'] == name].drop(["stock","y","uniqueDate","ret"],axis=1)
            #self.y_train_dic[name] = self.trainSet.loc[self.trainSet['stock'] == name]["y"]
            self.y_val_dic[name] = self.valSet.loc[self.valSet['stock'] == name]["y"]
            #self.return_train_dic[name] = self.trainSet.loc[self.trainSet['stock'] == name]["ret"]
            self.return_val_dic[name] = self.valSet.loc[self.valSet['stock'] == name]["ret"]


    def getEnsembleModelPCA(self, Cs, cv, features):
        acc_train_vec = np.zeros(self.N_stocks)
        acc_test_vec = np.zeros(self.N_stocks)
        for k, stock in enumerate(self.stocks):
            if round(100*(1+k)/self.N_stocks, 2) % 20 == 0:
                print('done %s%%'%round(100*(1+k)/self.N_stocks))
            name = stock + str(self.minutes_forward)
            X_train, y_train = self.X_train_dic[name][features],self.y_train_dic[name]
            X_test, y_test = self.X_test_dic[name][features], self.y_test_dic[name]
            try:
                m20 = LogisticClassifierPCA(pcaComponents=20)
                m20.fit(X_train,y_train)
                m30 = LogisticClassifierPCA(pcaComponents=30)
                m30.fit(X_train,y_train)
                m40 = LogisticClassifierPCA(pcaComponents=40)
                m40.fit(X_train,y_train)
                eclf = VotingClassifier(estimators=[('pca20', m20), ('pca30', m30), ('pca40', m40)], voting='soft', weights=None)
                eclf.fit(X_train,y_train)
                y_pred_train = eclf.predict(X_train)
                y_pred_test = eclf.predict(X_test)
                acc_train = 100 * (y_pred_train == y_train).mean()
                acc_test = 100 * (y_pred_test == y_test).mean()
                acc_train_vec[k]=acc_train
                acc_test_vec[k]=acc_test
                print(name[:-2],
                      'accuracy train', round(acc_train, 2),
                      'accuracy test', round(acc_test, 2))
            except np.linalg.linalg.LinAlgError as err:
                pass
        return acc_train_vec, acc_test_vec



# Class needed to apply the Voting Classifier to PCA
from sklearn.base import BaseEstimator, ClassifierMixin

class LogisticClassifierPCA(BaseEstimator, ClassifierMixin):
    def __init__(self, pcaComponents=20, Cs=np.logspace(-4, 5), cv=5, random_state=0):
        self.pcaComponents = pcaComponents
        self.pca = PCA(n_components=self.pcaComponents, random_state=random_state)
        self.Cs = Cs
        self.cv = cv
        self.random_state = random_state

    def fit(self, X, y=None):
        self.pca.fit(X)
        Xtransform = self.pca.transform(X)
        self.clf_ = LogisticRegressionCV(Cs=self.Cs, penalty='l2', cv=self.cv, random_state=random_state, refit=True)
        self.clf_.fit(Xtransform,y)
        return self
    
    def predict(self, X, y=None):
        try:
            getattr(self, "clf_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")
        Xtransform = self.pca.transform(X)
        return self.clf_.predict(Xtransform)

    def predict_proba(self, X, y=None):
        try:
            getattr(self, "clf_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")
        Xtransform = self.pca.transform(X)
        return self.clf_.predict_proba(Xtransform)

