from keras.layers import Input, Dense, Activation, Dropout
from keras.models import Model, Sequential
import keras.callbacks
from keras.regularizers import l1_l2
from keras import backend as K
import numpy as np
import os


class ModelType:
    Sequential, Merged, End2End = range(3)

class Autoencoder:

    def __init__(self, architecture, modelType=ModelType.Merged, weightsDirectory="_weights", dropout=0, inputNoise=0, l1reg=0, l2reg=0,autoencoderLoss='cosine_proximity'):
        # Create directory if does not exist
        if weightsDirectory is not None and not os.path.isdir(weightsDirectory):
            os.makedirs(weightsDirectory)
        self.wD = weightsDirectory
        self.architecture = architecture # Assumed of length 3 for now (3rd order interactions)
        self.modelType = modelType
        self.dropout = np.float32(dropout)
        self.inputNoise = np.float32(inputNoise)
        self.l1reg = l1reg
        self.l2reg = l2reg
        self.autoencoderLoss = autoencoderLoss
        self.freezed = False


    def buildAutoencoder(self,trainable=True):
        self.freezed = not trainable
        # Part common to all models
        input_img = Input(shape=( self.architecture[0] ,))
        encoded_0 = Dropout( self.inputNoise )(input_img) # Add noise to the input (using dropout) --> Denoising Autoencoder
        encoded_1 = Dense( self.architecture[1] , activation='relu', trainable=trainable)(encoded_0)
        encoded_1b = Dropout( self.dropout )(encoded_1) # Is still active when freezing the layers
        #encoded_2 = Dense( self.architecture[-1] , activation='relu', trainable=trainable, name="encoded")(encoded_1b)
        encoded_2 = Dense( self.architecture[-1] , trainable=trainable, name="encoded")(encoded_1b)
        
        # Decoding part
        if self.modelType == ModelType.Sequential or self.modelType == ModelType.Merged:
            decoded_1 = Dense( self.architecture[1] , activation='relu', trainable=trainable)(encoded_2)
            decoded_1b = Dropout( self.dropout )(decoded_1)
            decoded_2 = Dense( self.architecture[0] , trainable=trainable, name="decoded")(decoded_1b)
    
        # Logistic Regression
        reg = l1_l2(l1=self.l1reg, l2=self.l2reg)
        logit = Dense(2, input_dim= self.architecture[-1] , activation='softmax', kernel_regularizer=reg, name="logit")(encoded_2)
    
        # Define Model
        if self.modelType == ModelType.Sequential:
            self.model1 = Model(inputs=input_img, outputs=decoded_2)
        elif self.modelType == ModelType.Merged:
            self.model = Model(inputs=input_img, outputs=[decoded_2,logit])
        if self.modelType == ModelType.Sequential or self.modelType == ModelType.End2End:
            self.model = Model(inputs=input_img, outputs=logit)
    
        # Compile
        self.compileModel()
    

    def compileModel(self):
        adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        if self.modelType == ModelType.Sequential:
            self.model1.compile(optimizer=adam, loss={'decoded':self.autoencoderLoss}, metrics={'decoded':'mae'})
        elif self.modelType == ModelType.Merged:
            self.model.compile(optimizer=adam, loss={'decoded':self.autoencoderLoss,'logit':'categorical_crossentropy'},
                               loss_weights=[0.5,0.5],metrics={'decoded':'mae','logit':'acc'})
        if self.modelType == ModelType.Sequential or self.modelType == ModelType.End2End:
            self.model.compile(optimizer=adam, loss={'logit':'categorical_crossentropy'}, metrics={'logit':'acc'})

    # Freeze or Defreeze the autoencoder for learning but not the Logistic Regression
    def freezeAutoencoder(self,freeze):
        self.freezed=freeze
        self.buildAutoencoder(trainable=not freeze)
        if freeze:
            print("Freezing Model")
        else:
            print("Using Unfreezed Model")
        
        # Recompile
        self.compileModel()

    def fit(self, X_train, y_train, X_val, y_val, epochs=200, batch=256, checkpointName=None):
        # name = All 30 stocks _ (architecture) _ dropout _ input noise _ (regularisation) _ freezed
        self.name = "A30S_m%s_(%s,%s,%s)_d%s_in%s_r(%s,%s)_b%s_l(%s)_f%s"%( self.modelType,
                        self.architecture[0], self.architecture[1], self.architecture[2], self.dropout,
                        self.inputNoise, self.l1reg, self.l2reg, batch,
                        ("cp" if self.autoencoderLoss=='cosine_proximity' else "mse"), int(self.freezed) )
                        
        # Tensorboard not supported on Theano (so not on GPU)
        if K.backend() == 'tensorflow':
            tensorboard = keras.callbacks.TensorBoard(log_dir='log/%s'%(self.name),
                                                      histogram_freq=0, write_graph=True,
                                                      write_images=True)
        
        # Checkpoint to keep best model
        if self.modelType == ModelType.Merged:
            if self.freezed:
                monit='val_logit_loss'
            else:
                monit='val_decoded_loss'
        else:
            monit = 'val_loss'
        if checkpointName is None:
            filepath="%s/%s.hdf5"%(self.wD, self.name)
        else:
            filepath=checkpointName
        checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor=monit, verbose=1, save_best_only=True, mode='min')
        
        # Add early stopping
        earlystopping = keras.callbacks.EarlyStopping(monitor=monit, min_delta=0, patience=10, verbose=1, mode='auto')
        
        # Define callbacks
        if K.backend() == 'tensorflow':
            cbs = [tensorboard, checkpoint, earlystopping]
        else:
            cbs = [checkpoint, earlystopping]
        
        # Train
        if self.modelType == ModelType.Sequential and not self.freezed:
            print("\tTraining First part of Sequential Model")
            self.model1.fit(x=X_train, y=X_train, epochs=epochs, batch_size=batch, shuffle=True,
                            validation_data=(X_val,X_val), verbose=0, callbacks=cbs)

        elif self.modelType == ModelType.Merged:
            self.model.fit(x=X_train, y=[X_train,y_train], epochs=epochs, batch_size=batch, shuffle=True,
                           validation_data=(X_val,[X_val,y_val]), verbose=0, callbacks=cbs)
        
        elif self.modelType == ModelType.End2End or (self.modelType == ModelType.Sequential and self.freezed):
            if self.modelType == ModelType.Sequential:
                print("\tTraining Second part of Sequential Model --> Remember to freeze First Part")
            else:
                print("\tTraining End-to-End Model")
            self.model.fit(x=X_train, y=y_train, epochs=epochs, batch_size=batch, shuffle=True,
                           validation_data=(X_val,y_val), verbose=0, callbacks=cbs)


    # Load from weights
    def loadFromWeights(self, checkpointName):
        print(checkpointName)
        if self.modelType == ModelType.Sequential and "_f0_" in checkpointName:
            #print(self.model1.summary())
            self.model1.load_weights(checkpointName)
        else:
            #print(self.model.summary())
            self.model.load_weights(checkpointName)
        self.compileModel()

    def predict(self, X_test, withDecoding=False):
        # if dataset provided too large, extrat relevant part --> HARDCODED...
        input_dim = self.architecture[0]
        # Check the hardcoded assumptions
        assert(X_test.shape[1] == input_dim or ( X_test.shape[1] == 239 and (input_dim==119 or input_dim==120)))
        
        # Define final input set
        if X_test.shape[1] == input_dim:
            X = X_test
        elif input_dim == 119:
            X = X_test[:,:119]
        elif input_dim == 120:
            X = X_test[:,119:]
        
        prediction = self.model.predict(X)
        if self.modelType == ModelType.Merged and not withDecoding:
            return prediction[1]
        elif self.modelType == ModelType.Sequential and withDecoding:
            prediction1 = self.model1.predict(X)
            return [prediction, prediction1]
        else:
            return prediction



class AutoencoderEnsemble:
    def __init__(self,modelNames):
        self.modelNames = modelNames

    def loadModels(self, modelType, architecture, weightsDirectory=None):
        self.modelList = []
        for n in self.modelNames: #A30S_m1_(120,100,30)_d0.00684_in0.0698_r(0,0.005)_b256_l(cp)_f0.hdf5
            auto = Autoencoder(architecture, modelType, weightsDirectory) # don't care about training parameters (eg: dropout)
            auto.buildAutoencoder()
            if weightsDirectory is not None:
                chkpt_0 = "%s/%s.hdf5"%(weightsDirectory,n)
            else:
                chkpt_0 = "%s.hdf5"%n
            auto.loadFromWeights(chkpt_0)
            self.modelList.append(auto)

    def votes(self,probas):
        # probas is [nModels, nSamples, 2]
        hardVotingIndex = np.sum(np.argmax(probas,axis=2),axis=0)>(probas.shape[0]/2)
        hardVoting = np.zeros((hardVotingIndex.shape[0], 2))
        hardVoting[np.arange(hardVotingIndex.shape[0]), hardVotingIndex.astype('int32')] = 1

        softVoting = np.sum(probas,axis=0)/probas.shape[0]

        productVoting = np.product(probas,axis=0)
        productVoting = productVoting/np.repeat(np.sum(productVoting,axis=1,keepdims=True),2,axis=1)
        
        # all [1,0] --> [1,0]  |  all [0,1] --> [0,1]  |  else [0,0]
        unanimityVotingIndex1 = np.sum(np.argmax(probas,axis=2),axis=0)==0
        unanimityVotingIndex2 = np.sum(np.argmax(probas,axis=2),axis=0)==probas.shape[0]
        unanimityVoting = np.zeros((unanimityVotingIndex1.shape[0], 2))
        unanimityVoting[:,0] = unanimityVotingIndex1.astype('int32')
        unanimityVoting[:,1] = unanimityVotingIndex2.astype('int32')
        
        return hardVoting,softVoting,productVoting,unanimityVoting,probas

    def predict(self,X_test):
        probas = np.zeros([len(self.modelList),X_test.shape[0],2])
        for k,m in enumerate(self.modelList):
            probas[k,:,:] = m.predict(X_test)
        return self.votes(probas)




# Class used to store the results from an experiment
class ExperimentPerformance:
    def __init__(self, methodName, stocks, originalFeatures):
        self.methodName = methodName
        self.stocks = stocks
        self.originalFeatures = originalFeatures
    
    
    # vector of number of components used (can be ["10", "20", "30", "ensemble"])
    def setNumComponents(numComponents):
        self.numComponents = numComponents
    
    ###############################################################
    # results as a matrix (stocks, numComponents)
    def setTrainResults(self, trainResults, trainNumOfTrades):
        assert(trainNumOfTrades.shape[0]==len(self.stocks))
        self.trainResults = trainResults
        self.trainNumOfTrades = trainNumOfTrades
    
    def setValResults(self, valResults, valNumOfTrades):
        assert(valNumOfTrades.shape[0]==len(self.stocks))
        self.valResults = valResults
        self.valNumOfTrades = valNumOfTrades
    
    def setSuperValResults(self, supervalResults, supervalNumOfTrades):
        assert(supervalNumOfTrades.shape[0]==len(self.stocks))
        self.supervalResults = supervalResults
        self.supervalNumOfTrades = supervalNumOfTrades
    
    def setTestResults(self, testResults, testNumOfTrades):
        assert(testNumOfTrades.shape[0]==len(self.stocks))
        self.testResults = testResults
        self.testNumOfTrades = testNumOfTrades
    
    ###############################################################
    def getTestResults(self):
        return self.testResults
    
    def getMaxTestResults(self):
        if self.testResults.ndim == 1:
            return self.testResults
        else:
            return np.max(self.testResults,axis=1)

    ###############################################################
    # Compute accuracies
    def getAccuracy(self, set='test', avg=False):
        if set == 'test':
            res = self.testResults
            numTrades = self.testNumOfTrades
        elif set == 'val':
            res = self.valResults
            numTrades = self.valNumOfTrades
        elif set == 'train':
            res = self.trainResults
            numTrades = self.trainNumOfTrades
        elif set == 'superval':
            res = self.supervalResults
            numTrades = self.supervalNumOfTrades
        if not avg or res.ndim == 1:
            return np.sum(numTrades*res,axis=0)/np.sum(numTrades, axis=0)
        else:
            return np.mean(np.sum(numTrades*res,axis=0)/np.sum(numTrades, axis=0))

    def getAccuracyMinusSigma(self, set='test', avg=False):
        if set == 'test':
            numTrades = self.testNumOfTrades
            res = self.testResults
        elif set == 'val':
            numTrades = self.valNumOfTrades
            res = self.valResults
        elif set == 'train':
            numTrades = self.trainNumOfTrades
            res = self.trainResults
        elif set == 'superval':
            res = self.supervalResults
            numTrades = self.supervalNumOfTrades
        mean = np.sum(numTrades*res,axis=0)/np.sum(numTrades, axis=0)

        var = np.sum( numTrades * np.power( res - np.repeat( mean[None,], res.shape[0], axis=0 ), 2), axis = 0) / np.sum(numTrades, axis=0)
        
        if not avg or res.ndim == 1:
            return mean - np.sqrt(var)
        else:
            return np.mean(mean - np.sqrt(var))

