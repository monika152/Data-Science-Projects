from keras import models
from keras import layers
import tensorflow as tf

class Fit_model:
    def __init__(self, layer_1, layer_2, activ, X, epoch, X_train, y_train) -> None:
        self.layer_1 = layer_1
        self.layer_2 = layer_2
        self.activ = activ
        self.X = X
        self.epoch = epoch
        self.X_train = X_train
        self.y_train = y_train
    
        #Build the Network
        self.network = models.Sequential()  # linear stack of layers

        self.network.add(layers.Dense(self.layer_1, activation=self.activ[0], input_shape = (self.X.shape[1],))) # (x,) vector 
        self.network.add(layers.Dense(self.layer_2, activation=self.activ[0])) # the info about input_shape is taken from the layer above

        # Output Layer
        self.network.add(layers.Dense(1, activation=self.activ[0]))


        self.network.summary()  # Parameters: weight + bias


        #Complie / Configure the network
        self.network.compile(loss='binary_crossentropy', optimizer=tf.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])


        #Fit the Network
        self.history = self.network.fit(self.X_train, self.y_train, epochs=self.epoch, shuffle=True, verbose=1, validation_split= 0.09, batch_size=5)