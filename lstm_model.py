import tensorflow as tf
import numpy as np
from preprocess_prices import get_data

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()

        self.window_size = 15
        self.window_shift = 1

        # Initialize all hyperparameters
        self.learning_rate = 0.005
        self.num_epochs = 10
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        # Initialize all trainable parameters
        self.gru = tf.keras.layers.GRU(64, return_sequences=True, return_state=False)
    

    def call(self, inputs, initial_state):
        """
        Runs a forward pass on an input batch of time series data for a cryptocurrency.
        
        :param inputs:
        :param initial_state:
        :return: 
        """
        
        return self.gru(inputs, initial_state=initial_state)


    def loss(self, predictions, labels):
        """
        Calculates the model's mean squared error loss after one forward pass.
        
        :param predictions: 
        :param labels: 
        """

        return tf.reduce_mean(tf.keras.losses.mse(labels, predictions))

    def accuracy(self, predictions, labels):
        return tf.math.mean((predictions >= 0) == (labels >= 0))


def train(model, inputs, initial_state):
    '''
    Trains the model on all of the inputs for one epoch.
    
    :param model: 
    :param 
    :param 
    :return: List of losses for every 50th window
    '''

    losses = []

    for idx in range(0, tf.shape(inputs)[0] - model.window_size - 1, model.window_shift):
        window_inputs = inputs[idx:idx+model.window_size]
        window_labels = inputs[idx+1:idx+model.window_size+1]

        with tf.GradientTape() as tape:
            predictions = model(window_inputs, initial_state)
            loss = model.loss(predictions, window_labels)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if idx % 50 == 0:
            losses.append(loss)

    return losses


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
        window_labels = test_inputs[idx+1:idx+model.window_size+1]
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
    train_inputs, train_labels = get_data('../../data/train', 3, 5)
    test_inputs, test_labels = get_data('../../data/test', 3, 5)

    # Initialized model
    model = Model()

    # Trains model
    for _ in range(model.num_epochs):
        train(model, train_inputs, train_labels)

    # Tests the accuracy
    print("Test accuracy: " + str(np.round(test(model, test_inputs, test_labels).numpy() * 100, decimals=2)) + "%")


if __name__ == '__main__':
    main()
