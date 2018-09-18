from attention import Attention
import os
import numpy as np
import pickle
from data_generator import DataGenerator
import matplotlib.pyplot as plt
plt.switch_backend('GTKAgg') 
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, LSTM, Reshape, TimeDistributed, Input, Lambda, Concatenate
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import Bidirectional
from keras.models import Sequential, Model, load_model

from keras.optimizers import Adam
import model_parameters
from sklearn.metrics import precision_score,recall_score, f1_score
import tensorflow as tf
import functools as f

from itertools import product


class KerasNeuralNetwork():

    def __init__(self):
        self._model = None
    
    
     
   
   
    def model_(self):
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
      
 
   
    def model_2(self):
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
        past = Lambda(lambda x : x[:,:26,:])(dropout_embeddings)
        future = Lambda(lambda x : x[:,25:,:])(dropout_embeddings)
        past_LSTM = LSTM(config.token_embedding_dim * config.num_of_fields,  go_backwards=True, return_sequences=True)(past)
        future_LSTM = LSTM(config.token_embedding_dim * config.num_of_fields,  go_backwards=False, return_sequences=True)(future)
        #merged = Concatenate(axis=1)([past_LSTM, future_LSTM])
        attention_1 =  Attention()(past_LSTM)
        attention_2 = Attention()(future_LSTM)
        merged = Concatenate(axis=1)([attention_1, attention_2])
        dense1 = Dense(2000, activation='relu')(merged)
        dense2 = Dense(1000, activation='relu')(dense1)
        outputs = Dense(2, activation='softmax')(dense2)
        model = Model(inputs=inputs, outputs=outputs)

        model.compile(optimizer=Adam(lr=config.lr, clipvalue=5.0),
                      loss="hinge",
                      metrics=['binary_accuracy'])
        print(model.summary())

        return model


    def train(self, vocabulary, dictionaryOfFrequencies):
        config = model_parameters.train_config()
        
        training_generator = DataGenerator(range(96), config.input_directory_train, batch_size=config.batch_size)
        validation_generator = DataGenerator(range(96,100), config.input_directory_train, batch_size=config.batch_size)

        model = self.model()
        
        #checkpointing
        filepath= config.model_path+"-{epoch:02d}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=False, mode='min')
        callbacks_list = [checkpoint]

        steps_per_epoch = config.train_examples / config.batch_size
        validation_steps = config.validation_examples / config.batch_size
        class_weight ={0:10, 1:2} 
        print("Start model training")
        # fit the model
        fit_model_result = model.fit_generator(generator=training_generator.flow_from_directory(),
                                               validation_data=validation_generator.flow_from_directory(),
                                               steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, class_weight = class_weight, epochs=config.epochs,
                                               callbacks=[checkpoint])
        
    def test(self):
        precisions = []
        recalls = []
        f1_s = []
        model_dir = "kdd_models/"
        for f in os.listdir(model_dir):
            p =[]
            r =[]
            print(f)
            config = model_parameters.test_config()
            test_generator = DataGenerator(range(8), config.input_directory_test ,batch_size=config.batch_size, train = False)
        
            loaded_model = load_model(model_dir+f,custom_objects={"Attention": Attention})
            print(loaded_model.summary())
   
            # evaluate loaded model on test data
            loaded_model.compile(loss="hinge", optimizer=Adam())
            test_steps = (config.test_examples/config.batch_size)
            predict = loaded_model.predict_generator(test_generator.flow_from_directory(), steps=test_steps, verbose=1)
        
            y_true = []
            for i in range(8):
                y_true = np.append(y_true,np.load(config.input_directory_test+str(i)+'labels_test.npy'))

            print(y_true.shape)
            y = predict[:,-1] 
            
            if len(y) !=  len(y_true):
                diff = len(y_true)-len(y)
                y_true = y_true[:-diff]
            
            p, r, f1 = self.precision_recall_curves(y_true, y)
            print(f1)            
            print(p)
            print(r)
            precisions.append(p)
            recalls.append(r)
            f1_s.append(f1)
        np.save("kdd_models/precisions.npy", precisions)
        np.save("kdd_models/recalls.npy", recalls)
        np.save("kdd_models/f1.npy", f1_s)
        #self.precision_recall_plot(self, p, r)

        
    
        
    def precision_recall_curves(self, y_true, y):
        precision = []
        recall = []
        print(y_true[:10])
        print(y[:10])
        for threshold in np.linspace(0,1,11):
            if threshold==0: continue
            predictions = (y>threshold).astype(int)
            print(threshold)
            print(predictions[:10])
            recall.append(recall_score(y_true, predictions))
            precision.append(precision_score(y_true, predictions))
        f1 =  f1_score(y_true, predictions)
        return precision, recall, f1



    def precision_recall_plot(precisions, recalls):
        plt.gca().set_color_cycle(['red', 'green', 'blue', 'yellow','purple','black','magenta'])
        for i in range(5):
            plt.plot(recalls[i],precisions[i])
            
        plt.legend(['1 epoch', '2 epochs', '3 epochs', '4 epochs','5 epochs', '6 epochs', '7 epochs', '8 epochs', '9 epochs','10 epochs'], loc='upper right')
        
        plt.step(recalls, precisions, color='b', alpha=0.2,where='post')
        #plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.show()
        plt.title('Precision-Recall curve')



    
    def main(self, train=False):
        config = model_parameters.train_config()
        
        with open(config.vocabulary_dir, 'rb') as handle:
            vocabulary = pickle.load(handle)


        with open(config.frequency_dict_dir, 'rb') as handle:
            dictionaryOfFrequencies = pickle.load(handle)
            
        print("Vocabulary and dictionary of frequencies loaded.")
        
        if (train):
         print("Training starts")
         self.train(vocabulary, dictionaryOfFrequencies)
        else: self.test()


if __name__ == '__main__':
    nn = KerasNeuralNetwork()
    nn.main()
