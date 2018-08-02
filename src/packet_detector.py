from attention import Attention
import csv
from data_generator import DataGenerator
import glob
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, LSTM, Reshape, TimeDistributed, Input
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import Bidirectional
from keras.models import Sequential, Model, model_from_json
from keras.optimizers import Adam
import model_parameters
import numpy as np
import os
import pickle
from random import randint, random
from sklearn.metrics import precision_recall_curve
import tensorflow as tf


class KerasNeuralNetwork():

    def __init__(self):
        self._model = None



 
 #==============================================================================
 # 
 #    def model(self):
 #        model = Sequential()
 #        model.add(Embedding(VOCAB_SIZE, EMB_DIMENSION, input_length=(NUM_OF_LOG_FIELDS * TIME_STEPS)))
 #        model.add(Reshape((TIME_STEPS, (EMB_DIMENSION * NUM_OF_LOG_FIELDS)),
 #                          input_shape=((NUM_OF_LOG_FIELDS * TIME_STEPS), EMB_DIMENSION)))
 #        model.add(Dropout(DROPOUT_RATE))
 #        model.add(Bidirectional(LSTM(EMB_DIMENSION * NUM_OF_LOG_FIELDS, return_sequences=True)))
 #        model.add(Dropout(DROPOUT_RATE))
 #        model.add(TimeDistributed(Dense(1000, activation='softmax')))
 #        model.add(TimeDistributed(Dense(500, activation='softmax')))
 #        model.add(TimeDistributed(Dense(2, activation='softmax')))
 # 
 #        model.compile(optimizer=Adam(lr=lr, clipvalue=5.0),
 #                      loss="binary_crossentropy",
 #                      metrics=['binary_accuracy'])
 #        print(model.summary())
 # 
 #        return model
 #     
 #==============================================================================



    def model(self):
        config = model_parameters.model_config()
        model = Sequential()
        model.add(Embedding(config.vocab_size, config.token_embedding_dim, input_length=(config.num_of_fields * config.time_steps)))
        model.add(Reshape((config.time_steps, (config.token_embedding_dim * config.num_of_fields)),
                          input_shape=((config.num_of_fields * config.time_steps), config.token_embedding_dim)))
        model.add(Dropout(config.dropout_rate))
        
        model.add(Bidirectional(LSTM(config.token_embedding_dim * config.num_of_fields , return_sequences=True)))
        model.add(Attention())
        model.add(Dense(2, activation='softmax'))

        model.compile(optimizer=Adam(lr=config.lr, clipvalue=5.0),
                      loss=config.loss,
                      metrics=['binary_accuracy'])
        print(model.summary())

        return model
    


    def train(self, vocabulary, dictionaryOfFrequencies):
        
        config = model_parameters.train_config()
        
        training_generator = DataGenerator(range(96), config.input_directory_train, batch_size=config.batch_size)
        validation_generator = DataGenerator(range(96, 100), config.input_directory_train, batch_size=config.batch_size)

        model = self.model()
        # checkpoint
        checkpoint = ModelCheckpoint(config.weights_path, monitor='val_acc', verbose=1, save_best_only=False,
                                     save_weights_only=False, mode='auto', period=1)
        callbacks_list = [checkpoint]

        steps_per_epoch = config.train_examples / config.batch_size
        validation_steps = config.validation_examples / config.batch_size
        
        print("Start model training")
        # fit the model
        fit_model_result = model.fit_generator(generator=training_generator.flow_from_directory(),
                                                validation_data=validation_generator.flow_from_directory(),
                                                steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, epochs=config.epochs)

        # serialize model to JSON
        print("Saving model to disk.")
        model_json = model.to_json()
        with open(config.model_path, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(config.weights_path)
        print("Saved model to disk")

        return model
    
    
    

    def test(self, vocabulary, dictionaryOfFrequencies):
  
        config = model_parameters.test_config()
        test_generator = DataGenerator(range(5), batch_size=BATCH_SIZE)   
        loaded_model = self.load_saved_model(config.model_path, config.weights_path)

        precision = self.as_keras_metric(tf.metrics.precision)
        recall = self.as_keras_metric(tf.metrics.recall)

        # evaluate loaded model on test data
        loaded_model.compile(loss=self.weighted_categorical_crossentropy([0.3, 0.8]), optimizer=Adam(),
                             metrics=['accuracy'] + 
                                     [precision, recall] + 
                                     [self.precision_threshold(i) for i in np.linspace(0.1, 0.9, 9)] + 
                                     [self.recall_threshold(i) for i in np.linspace(0.1, 0.9, 9)])

        metrics = loaded_model.evaluate_generator(test_generator.flow_from_directory(), steps=test_examples / BATCH_SIZE, verbose=VERBOSE)
        print('Accuracy: %f' % (metrics[1] * 100))
        print('F1: %f' % (self.f1_measure(metrics[3], metrics[2]) * 100))
        print('Recall: %f' % (metrics[2] * 100))
        print('Precision: %f' % (metrics[3] * 100))

        print("Precision over different thresholds")
        print(metrics[4:12])
        print("Recall over different thresholds")
        print(metrics[13:])
        
        predict = loaded_model.predict_generator(test_generator.flow_from_directory(), steps=test_examples / BATCH_SIZE, verbose=VERBOSE)
        with open('output_predictions.txt', 'w') as f:
            for _list in predict:
                for _string in _list:
                    f.write(str(_string) + '\n')

  


    def as_keras_metric(self, metric):
        import functools

        @functools.wraps(metric)
        def wrapper(self, args, **kwargs):
            """ Wrapper for turning tensorflow metrics into keras metrics """
            value, update_op = metric(self, args, **kwargs)
            K.get_session().run(tf.local_variables_initializer())
            with tf.control_dependencies([update_op]):
                value = tf.identity(value)
            return value

        return wrapper
    

    def load_saved_model(self, model_path, weights_path):
        # load json and create model
        json_file = open(model_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(weights_path)
        print("Loaded model from disk")
        return loaded_model
    

    def f1_measure(self, precision, recall):
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

    def precision_recall_plot(self, y_true, y_pred):
        import tensorflow as tf
        y_true = tf.keras.backend.eval(y_true)
        y_pred = tf.keras.backend.eval(y_pred)
        print(y_true)
        print(y_pred)
        precision, recall, threashold = precision_recall_curve(y_true, y_pred)

        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        # plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))

    def precision_threshold(self, threshold=0.5):
        def precision(y_true, y_pred):
            """Precision metric.
            Computes the precision over the whole batch using threshold_value. """
            threshold_value = threshold
            y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
            true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
            # predicted positives = true positives + false positives
            predicted_positives = K.sum(y_pred)
            precision_ratio = true_positives / (predicted_positives + K.epsilon())
            return precision_ratio

        return precision

    def recall_threshold(self, threshold=0.5):
        def recall(y_true, y_pred):
            """Recall metric. Computes the recall over the whole batch using threshold_value."""
            threshold_value = threshold
            y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
            true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
            # all positives = true positives + false negatives, false negatives einai ayta poy enw htan intrusions, ta
            all_positives = K.sum(K.clip(y_true, 0, 1))
            recall_ratio = true_positives / (all_positives + K.epsilon())
            return recall_ratio

        return recall

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
        
        if (train): self.train(vocabulary, dictionaryOfFrequencies)
        else: self.test(vocabulary, dictionaryOfFrequencies)


if __name__ == '__main__':
    nn = KerasNeuralNetwork()
    nn.main()
