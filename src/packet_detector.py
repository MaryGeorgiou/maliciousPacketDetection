from __future__ import print_function, division

from attention import Attention
import numpy as np
import pickle
from data_generator import DataGenerator
import model_parameters

import matplotlib as mpl
mpl.use('GTKAgg')
import matplotlib.pyplot as plt

from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, LSTM, Reshape, TimeDistributed, Input, Lambda, Concatenate
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import Bidirectional
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
from sklearn.metrics import precision_score,recall_score, f1_score




class KerasNeuralNetwork():

    def __init__(self):
        self._model = None
        
        
        
    def model_3(self):
        config = model_parameters.model_config()
        model = Sequential()
        model.add(Embedding(config.vocab_size, config.token_embedding_dim, input_length=(config.num_of_fields * config.time_steps)))
        model.add(Reshape((config.time_steps, (config.token_embedding_dim * config.num_of_fields)),
                          input_shape=((config.num_of_fields * config.time_steps), config.token_embedding_dim)))
        model.add(Dropout(config.dropout_rate))
        model.add(Bidirectional(LSTM(config.token_embedding_dim * config.num_of_fields, return_sequences=True)))
        model.add(Dropout(config.dropout_rate))
        model.add(TimeDistributed(Dense(1000, activation='softmax')))
        model.add(TimeDistributed(Dense(500, activation='softmax')))
        model.add(TimeDistributed(Dense(2, activation='softmax')))
   
        model.compile(optimizer=Adam(lr=config.lr, clipvalue=5.0),
                      loss="binary_crossentropy",
                      metrics=['binary_accuracy'])
        print(model.summary())
   
        return model
      
 
   
    def model_4(self):
        config = model_parameters.model_config()
        model = Sequential()
        model.add(Embedding(config.vocab_size, config.token_embedding_dim, input_length=(config.num_of_fields * config.time_steps)))
        model.add(Reshape((config.time_steps, (config.token_embedding_dim * config.num_of_fields)),
                          input_shape=((config.num_of_fields * config.time_steps), config.token_embedding_dim)))
        model.add(Dropout(config.dropout_rate))
        model.add(Bidirectional(LSTM(config.token_embedding_dim * config.num_of_fields, return_sequences=True)))
        model.add(Dropout(config.dropout_rate))
        model.add(Attention())
        model.add(Dense(2, activation='softmax'))
    
        model.compile(optimizer=Adam(lr=config.lr, clipvalue=5.0),
                      loss=self.weighted_categorical_crossentropy([0.7,0.3]),
                      metrics=['binary_accuracy'])
        print(model.summary())
    
        return model


    
    
    def model(self):
        config = model_parameters.model_config()
        inputs = Input(shape=(config.num_of_fields * config.time_steps,))
        embeddings = Embedding(config.vocab_size, config.token_embedding_dim, 
                               input_length=(config.num_of_fields * config.time_steps))(inputs)
        reshape = Reshape((config.time_steps, (config.token_embedding_dim * config.num_of_fields)),
                          input_shape=((config.num_of_fields * config.time_steps), config.token_embedding_dim))(embeddings)
        dropout_embeddings = Dropout(config.dropout_rate)(reshape)
        past = Lambda(lambda x : x[:25+1], output_shape=(26,1640))(dropout_embeddings)
        future = Lambda(lambda x : x[25:])(dropout_embeddings)
        past_LSTM = LSTM(config.token_embedding_dim * config.num_of_fields,  go_backwards=True, return_sequences=True)(past)
        future_LSTM = LSTM(config.token_embedding_dim * config.num_of_fields,  go_backwards=False, return_sequences=True)(future)
        merged = Concatenate([past_LSTM, future_LSTM])
        attention =  Attention()(merged)
        outputs = Dense(2, activation='softmax')(attention)
        model = Model(inputs=inputs, outputs=outputs)
 
        model.compile(optimizer=Adam(lr=config.lr, clipvalue=5.0),
                      loss=self.weighted_categorical_crossentropy([0.7,0.3]),
                      metrics=['binary_accuracy'])
        print(model.summary())
 
        return model




    def train(self, vocabulary, dictionaryOfFrequencies):
        config = model_parameters.train_config()
        
        training_generator = DataGenerator(range(96), config.input_directory_train, batch_size=config.batch_size)
        validation_generator = DataGenerator(range(96,101), config.input_directory_train, batch_size=config.batch_size)

        model = self.model()
        
        #checkpointing
        filepath= config.model_path+"-{epoch:02d}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]

        steps_per_epoch = config.train_examples / config.batch_size
        validation_steps = config.validation_examples / config.batch_size
        
        print("Start model training")
        # fit the model
        fit_model_result = model.fit_generator(generator=training_generator.flow_from_directory(),
                                               validation_data=validation_generator.flow_from_directory(),
                                               steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, epochs=config.epochs,
                                               callbacks=[checkpoint])
    
    


    def test(self):
        config = model_parameters.test_config()
        test_generator = DataGenerator(range(2),'data_test/' ,batch_size=config.batch_size)
        
        loaded_model = load_model('model_attention_keras.json-01.hdf5')
        print(loaded_model.summary())
   
        # evaluate loaded model on test data
        loaded_model.compile(loss="categorical_crossentropy", optimizer=Adam())
        test_steps = config.test_examples/config.batch_size      
        predict = loaded_model.predict_generator(test_generator.flow_from_directory(), steps=test_steps+1, verbose=1)
        
        y_true = []
        for i in range(2):
            y_true = np.append(y_true,np.load('data_test/'+str(i)+'labels_middle_test.npy'))

        predict = predict[:, 1] 
        p, r = self.precision_recall_curves(y_true, predict)
        self.precision_recall_plot(self, p, r)

        
    
        
    def precision_recall_curves(self, y_true, y):
        precision = []
        recall = []
        for threshold in np.linspace(0,1,11):
            predictions = (y>threshold).astype(int)
            recall.append(recall_score(y_true, predictions))
            precision.append(precision_score(y_true, predictions))
        f1 =  f1_score(y_true, predictions)
        return precision, recall, f1




    def precision_recall_plot(self, precision, recall):
        plt.step(recall, precision, color='b', alpha=0.2,where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.show()
        plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(np.average(precision)))



    def weighted_categorical_crossentropy(self, weights):
        weights = K.variable(weights)

        def loss(y_true, y_pred):
            # scale predictions so that the class probas of each sample sum to 1
            y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
            # clip to prevent NaN's and Inf's
            y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
            # calc
            loss = y_true * K.log(y_pred) * weights
            loss = -K.sum(loss, -1)
            return loss
        return loss
    
    
    def main(self, train=True):
        config = model_parameters.train_config()
        
        with open(config.vocabulary_dir, 'rb') as handle:
            vocabulary = pickle.load(handle)


        with open(config.frequency_dict_dir, 'rb') as handle:
            dictionaryOfFrequencies = pickle.load(handle)
            
        print("Vocabulary and dictionary of frequencies loaded.")
        
        if (train):
            print("Training starts")
            self.train(vocabulary, dictionaryOfFrequencies)
        else: 
            print("Testing")
            self.test(vocabulary, dictionaryOfFrequencies)




if __name__ == '__main__':
    nn = KerasNeuralNetwork()
    nn.main()
