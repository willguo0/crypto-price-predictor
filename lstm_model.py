import tensorflow as tf
import numpy as np
from preprocess_prices import get_data
import os

accs = []

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()

        self.window_size = 15
        self.window_shift = 1
        self.batch_size = 30

        # Initialize all hyperparameters
        self.learning_rate = 0.005
        self.num_epochs = 10
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        # Initialize all trainable parameters
        self.dense1 = tf.keras.layers.Dense(16)
        self.dense2 = tf.keras.layers.Dense(1)
        self.gru = tf.keras.layers.LSTM(64, return_sequences=True, return_state=False)
        self.dense3 = tf.keras.layers.Dense(64)
        self.dense4 = tf.keras.layers.Dense(1)
    

    def call(self, inputs, initial_state):
        """
        Runs a forward pass on an input batch of time series data for a cryptocurrency.
        
        :param inputs: shape [window_size, 3]
        :param initial_state:
        :return: 
        """
        
        fully_connected_output = tf.squeeze(self.dense2(self.dense1(tf.convert_to_tensor(inputs))))
        gru_output = self.gru(tf.reshape(fully_connected_output, [self.batch_size, self.window_size, 1]), initial_state=initial_state)
        return self.dense4(self.dense3(gru_output))


    def loss(self, predictions, labels):
        """
        Calculates the model's mean squared error loss after one forward pass.
        
        :param predictions: 
        :param labels: 
        """

        return tf.reduce_mean(tf.keras.losses.mse(labels, predictions))

    def accuracy(self, predictions, labels):
        return tf.math.reduce_mean(tf.cast((predictions >= 0) == (labels >= 0), tf.float32))


def train(model, inputs, initial_state):
    '''
    Trains the model on all of the inputs for one epoch.
    
    :param model: 
    :param 
    :param 
    :return: List of losses for every 50th window
    '''

    window_inputs = []

    for idx in range(0, tf.shape(inputs)[0] - model.window_size - 1, model.window_shift):
        window_inputs.append(inputs[idx:idx+model.window_size])

    losses = []

    for batch_idx in range(0, len(window_inputs) - model.batch_size - 1, model.batch_size):
        batch_inputs = window_inputs[batch_idx:batch_idx+model.batch_size]
        batch_labels = np.asarray(window_inputs[batch_idx+1:batch_idx+model.batch_size+1])[:, :, 0:1]

        with tf.GradientTape() as tape:
            predictions = model(batch_inputs, initial_state)
            loss = model.loss(predictions, batch_labels)
    
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        losses.append(loss)
        #print(loss)
        accs.append(model.accuracy(predictions, batch_labels))

    return tf.convert_to_tensor(losses)


def test(model, test_inputs, initial_state):
    """
    Tests the model on the test inputs and labels. You should NOT randomly 
    flip images or do any extra preprocessing.
    
    :param test_inputs: test data (all images to be tested), 
    shape (num_inputs, width, height, num_channels)
    :param test_labels: test labels (all corresponding labels),
    shape (num_labels, num_classes)
    :return: test accuracy - this should be the average accuracy across
    all batches
    """

    accuracies = []

    for idx in range(0, tf.shape(test_inputs)[0] - model.window_size - 1, model.window_shift):
        window_inputs = test_inputs[idx:idx+model.window_size]
        window_labels = np.asarray(test_inputs[idx+1:idx+model.window_size+1])[:, :, 0:1]
        predictions = model(window_inputs, initial_state)
        accuracies.append(model.accuracy(predictions, window_labels))
        
    return np.mean(accuracies)


def main():
    '''
    Read in CIFAR10 data (limited to 2 classes), initialize your model, and train and 
    test your model for a number of epochs. We recommend that you train for
    10 epochs and at most 25 epochs. 
    
    :return: None
    '''

    # Read in CIFAR10 data
    train_inputs, train_labels = get_data('./data/ada-usd-max.csv')
    
    directory = './data'
    crypto_data = {}
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f): 
           data, name = get_data(f)
           crypto_data[name] = data
        # Initialized model
    model = Model()

    # Trains model
    for crypto, data in crypto_data.items():
        print("Training " + crypto)
        p = train(model, data, None)
        print(p)

    print(tf.math.reduce_mean(accs))
    


if __name__ == '__main__':
    main()
