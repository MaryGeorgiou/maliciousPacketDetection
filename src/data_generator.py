import csv
import keras
import numpy as np


class DataGenerator():
    'Generates data for Keras'
    def __init__(self, IDs, data_dir, batch_size=128, n_classes=2):
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.IDs = IDs
        self.data_dir = data_dir


    def flow_from_directory(self):
        while True:
            X = []
            Y = []
            inputs = np.empty(shape=())
            targets = np.empty(shape=())
            for id in self.IDs:
                with open(self.data_dir + str(id) + 'kdd_indexed_test.csv', 'r') as csvfile:
                    X = []
                    Y = []
                    X = list(csv.reader(csvfile))
                    Y = keras.utils.to_categorical(np.load(self.data_dir + str(id) + 'labels_test.npy'), num_classes=self.n_classes)
                    print("File {} loaded.. Batching starts..".format(id)) 
                    b = 0 
                    inputs = np.empty(shape=())
                    targets = np.empty(shape=())
                    for i in xrange(0, len(X), self.batch_size): 
                        inputs = np.array(X[i:i + self.batch_size])
                        targets = np.array(Y[i:i + self.batch_size])
                        if b % 200 == 0: print("I'm in batch {}".format(b))
                        b = b + 1
                        yield inputs, targets
