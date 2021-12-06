import tensorflow as tf
import numpy as np
from preprocess_prices import get_data
import random
import os

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
        self.lstm = tf.keras.layers.LSTM(64, return_sequences=True, return_state=False)
        self.dense3 = tf.keras.layers.Dense(64)
        self.dense4 = tf.keras.layers.Dense(1)
    

    def call(self, inputs, initial_state):
        """
        Runs a forward pass on an input batch of time series data for a cryptocurrency.
        
        :param inputs: shape [window_size, 3]
        :param initial_state:
        :return: 
        """

        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        num_batches = inputs.shape[0]
        
        fully_connected_output = tf.squeeze(self.dense2(self.dense1(inputs)))
        lstm_output = self.lstm(tf.reshape(fully_connected_output, [num_batches, self.window_size, 1]), initial_state=initial_state)
        return self.dense4(self.dense3(lstm_output))


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
        batch_labels = np.asarray(window_inputs[batch_idx+1:batch_idx+model.batch_size+1], dtype=np.float32)[:, :, 0:1]

        with tf.GradientTape() as tape:
            predictions = model(batch_inputs, initial_state)
            loss = model.loss(predictions, batch_labels)
    
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        losses.append(loss)

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
        window_inputs = tf.expand_dims(test_inputs[idx:idx+model.window_size], axis=0)
        window_labels = tf.expand_dims(np.asarray(test_inputs[idx+1:idx+model.window_size+1], dtype=np.float32)[:, 0:1], axis=0)
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
    
    directory = './data'
    crypto_train_data = {}
    crypto_test_data = {}
    cryptos = []
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f): 
           data, name = get_data(f)
           train_data = data[0:-50]
           test_data = data[-50:]
           crypto_train_data[name] = train_data
           crypto_test_data[name] = test_data
           cryptos.append(name)
        # Initialized model
    model = Model()

    shuffled_cryptos = cryptos.copy()

    # Trains model
    for epoch in range(10):
        print("\nEpoch {}/10".format(epoch + 1))
        random.shuffle(shuffled_cryptos)
        for crypto in shuffled_cryptos:
            print(" - Training " + crypto)
            train(model, crypto_train_data[crypto], None)

    print("\nTest results:")
    print("\n{:<10} {:<10}".format('Crypto', 'Accuracy'))
    
    for crypto in cryptos:
        accuracy = test(model, crypto_test_data[crypto], None)
        print("{:<10} {:<10}".format(crypto, "{:.2%}".format(accuracy)))

if __name__ == '__main__':
    main()
