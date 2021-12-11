import tensorflow as tf
import numpy as np
from tensorflow.python.ops.gen_batch_ops import batch
from preprocess_prices import get_data
from preprocess_tweets import get_average_sentiment
from scipy.stats import binom
import random
import os

"""LSTM model that trains on cryptocurrency data"""
class Model(tf.keras.Model):
    def __init__(self, model_type):
        super(Model, self).__init__()

        assert(model_type in ['REGRESSION', 'CLASSIFICATION'])

        self.window_size = 64
        self.window_shift = 2
        self.batch_size = 64
        self.num_lstm_units = 128
        self.model_type = model_type
        self.num_epochs = 5

        # Initialize all trainable parameters
        self.dense1 = tf.keras.layers.Dense(32, activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1))
        self.dense2 = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1))
        self.init_hidden_state = tf.Variable(tf.random.truncated_normal([1, self.num_lstm_units], stddev=0.1, dtype=tf.float32))
        self.lstm = tf.keras.layers.LSTM(self.num_lstm_units, return_sequences=True, return_state=False)
        self.dropout1 = tf.keras.layers.Dropout(0.4)
        self.dense3 = tf.keras.layers.Dense(256, activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1))
        self.dropout2 = tf.keras.layers.Dropout(0.3)
        self.dense4 = tf.keras.layers.Dense(128, activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1))
        self.dropout3 = tf.keras.layers.Dropout(0.2)
        self.final_layer = tf.keras.layers.Dense(1)
        self.initial_state_dense = tf.keras.layers.Dense(self.num_lstm_units, activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1))

        
        if model_type == 'REGRESSION':
            self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-3, decay_steps=512, decay_rate=0.7)
        else:
            self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-2, decay_steps=512, decay_rate=0.8)
            

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)
    

    def call(self, inputs, init_cell_state, is_training=False):
        """
        Runs a forward pass on an input batch of time series data for a cryptocurrency.
        
        :param inputs: shape [window_size, 3]
        :param initial_state: initial state of the LSTM, includes twitter data as well as market cap and volume
        :return: 
        """

        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        init_cell_state = tf.convert_to_tensor(init_cell_state, dtype=tf.float64)
        init_cell_state = tf.reshape(init_cell_state, [1, -1])
        init_cell_state = self.initial_state_dense(init_cell_state)
        num_batches = inputs.shape[0]
        dense1_output = self.dense1(inputs)
        dense2_output = tf.squeeze(self.dense2(dense1_output))
        batch_initial_hidden_state = tf.repeat(self.init_hidden_state, repeats=num_batches, axis=0)
        batch_initial_cell_state = tf.repeat(init_cell_state, repeats=num_batches, axis=0)
        lstm_output = self.lstm(tf.reshape(dense2_output, [num_batches, self.window_size, 1]), initial_state=[batch_initial_hidden_state, batch_initial_cell_state])
        dense3_output = self.dense3(self.dropout1(lstm_output, training=is_training))
        dense4_output = self.dense4(self.dropout2(dense3_output, training=is_training))
        return self.final_layer(self.dropout3(dense4_output, training=is_training))

    def loss(self, predictions, labels, balance_preds=True, func='mse', ignore_correct_ratio=0.5):
        """
        Takes in the predicted percent changes, and either takes the mean standard error or mean average error.

        :param inputs: predictions  
        :return: scalar loss

        """
        assert(func in ['mse', 'mae'])
        assert(ignore_correct_ratio >= 0 and ignore_correct_ratio <= 1)
        

        loss = 0

        if balance_preds:
            mean = tf.reduce_mean(predictions)
            loss += tf.square(mean - (0.5 if self.model_type == 'CLASSIFICATION' else 0)) * 5

        if self.model_type == 'CLASSIFICATION':
            tf.keras.losses.mae
            loss += tf.reduce_mean(tf.keras.losses.mae(tf.cast(labels >= 0, dtype=tf.float32), predictions))
            return loss

        if ignore_correct_ratio > 0:
            correct_mask = tf.cast((predictions >= 0) == (labels >= 0), dtype=tf.float32)
            ignore_mask = 1 - correct_mask * ignore_correct_ratio
            labels *= ignore_mask
            predictions *= ignore_mask

        if func == 'mse':
            loss += tf.reduce_mean(tf.keras.losses.mse(labels, predictions))
        else:
            loss += tf.reduce_mean(tf.keras.losses.mae(labels, predictions))

        return loss

    def accuracy(self, predictions, labels):
        if self.model_type == 'CLASSIFICATION':
            return tf.math.reduce_mean(tf.cast((predictions >= 0.5) == (labels >= 0), tf.float32))
        else:
            return tf.math.reduce_mean(tf.cast((predictions >= 0) == (labels >= 0), tf.float32))


def train(model, window_inputs, init_cell_state, cryptocurrency_types):
    '''
    Trains the model on all of the inputs for one epoch.
    
    :param model: 
    :param window_inputs:
    :param init_cell_state:
    :return: losses, accuracy, mean of predictions, standard deviation of predictions
    '''

    losses = []
    accuracies = []
    all_predictions = []

    shuffled_window_inputs = window_inputs.copy()
    shuffled_cryptocurrency_types = cryptocurrency_types.copy()
    init_cell_state = tf.convert_to_tensor(init_cell_state)
    size = len(window_inputs)
    indexes = np.arange(0, size)

    indexes = tf.random.shuffle(indexes)
    
    shuffled_window_inputs = tf.gather(shuffled_window_inputs,indexes,axis=0)
    shuffled_cryptocurrency_types = tf.gather(shuffled_cryptocurrency_types,indexes,axis=0)
    
    for batch_idx in range(0, len(window_inputs) - model.batch_size - 1, model.batch_size):
        batch_inputs = np.asarray(shuffled_window_inputs)[batch_idx:batch_idx+model.batch_size, :-1, :]
        batch_labels = np.asarray(shuffled_window_inputs)[batch_idx:batch_idx+model.batch_size, 1:, 0:1]
        batch_cryptocurrencies = np.asarray(shuffled_cryptocurrency_types)[batch_idx:batch_idx+model.batch_size]
        batch_cryptocurrencies = tf.convert_to_tensor(batch_cryptocurrencies, dtype=tf.int32)
        batch_cryptocurrencies = tf.reshape(batch_cryptocurrencies, [-1,1])
        curr_sentiment = tf.gather_nd( init_cell_state,batch_cryptocurrencies,batch_dims = 0)
        
        with tf.GradientTape() as tape:
            predictions = model(batch_inputs, curr_sentiment, is_training=True)
            loss = model.loss(predictions, batch_labels)

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        losses.append(loss)
        accuracies.append(model.accuracy(predictions, batch_labels))
        all_predictions.append(predictions)

    return tf.reduce_mean(losses), tf.reduce_mean(accuracies), tf.reduce_mean(all_predictions), tf.math.reduce_std(all_predictions)


def test(model, test_inputs, init_cell_state):
    """
    Tests the model on the test inputs and labels.
    
    :param test_inputs: test data, 
    :param init_cell_state: initial state to set LSTM,
    :param test_labels: test labels (all corresponding labels),
    :return: test accuracy, total number of predictions made
    """

    accuracies = []
    num_predictions = 0
    init_cell_state = [init_cell_state] * 64
    for idx in range(0, tf.shape(test_inputs)[0] - model.window_size - 1, model.window_size):
        window_inputs = tf.expand_dims(test_inputs[idx:idx+model.window_size], axis=0)
        window_labels = tf.expand_dims(np.asarray(test_inputs[idx+1:idx+model.window_size+1], dtype=np.float32)[:, 0:1], axis=0)
        predictions = model(window_inputs, init_cell_state, is_training=False)
        num_predictions += tf.size(predictions).numpy()
        accuracies.append(model.accuracy(predictions, window_labels))
        
    return np.mean(accuracies), num_predictions


def main():
    '''
    Read in crypto price and twitter data, initializes model, and trains and 
    tests model for a number of epochs.
    
    :return: None
    '''
    model = Model(model_type="CLASSIFICATION")
    directory = './data/price-data'
    crypto_train_data = []
    crypto_test_data = {}
    cryptos = []
    sentiment = []
    window_train_data = []
    window_train_types = []
    for i, filename in enumerate(os.listdir(directory)):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f): 
<<<<<<< HEAD
            data, name = get_data(f)
            train_data = data[:-66]
            test_data = data[-66:]
            crypto_train_data.append(train_data)
            crypto_test_data[name] = test_data
            cryptos.append(name)
            sentiment.append(get_average_sentiment(name))

            for j in range(0, tf.shape(train_data)[0] - model.window_size - 1, model.window_shift):
                window_train_data.append(train_data[j:j+model.window_size+1])
                window_train_types.append(i)
=======
           data, name = get_data(f)
           train_data = data[:-100]
           test_data = data[-100:]
           crypto_train_data.append(train_data)
           crypto_train_types += [idx] * train_data.shape[0]
           crypto_test_data[name] = test_data
           cryptos.append(name)
>>>>>>> 3ca836dd5cbabcf52ccbacd1f517d3fd6840ee2f
        # Initialized model
    
    crypto_train_data = np.row_stack(crypto_train_data)    

    

    print("="*67)
    print("Training".center(67))
    print("="*67)
    print()
    print(" {:<10}|{:<10}|{:<10}|{:<10}|{:<10}|{:<10}".format("Epoch".center(10), "Loss".center(10), "Accuracy".center(10), "Mean".center(10), "Std".center(10), "LR".center(10)))
    print(" {:<10}|{:<10}|{:<10}|{:<10}|{:<10}|{:<10}".format("-"*10, "-"*10, "-"*10, "-"*10, "-"*10, "-"*10))

    # Trains model
    for epoch in range(model.num_epochs):
        loss, accuracy, mean, std = train(model, window_train_data, sentiment, window_train_types)
        # loss, accuracy, mean, std = train(model, window_train_data, tf.zeros([1, model.num_lstm_units]))
        print(" {:<10}|{:<10}|{:<10}|{:<10}|{:<10}|{:<10}".format(epoch + 1, "{:.4f}".format(loss), "{:.2%}".format(accuracy), "{:.4f}".format(mean), "{:.4f}".format(std), "{:.4f}".format(model.optimizer._decayed_lr(var_dtype=tf.float32))))

    print()
    print("="*67)
    print("Testing".center(67))
    print("="*67)
    print()
    print("{:<10}|{:<10}|{:<10}".format("Crypto".center(10), "Accuracy".center(10), "P-value".center(10)).center(67))
    print("{:<10}|{:<10}|{:<10}".format("-"*10, "-"*10, "-"*10).center(67))

    total_examples = 0
    total_correct = 0
    i = 0
    for crypto in cryptos:
        accuracy, num_examples = test(model, crypto_test_data[crypto], get_average_sentiment(crypto))
        i = i + 1
        # accuracy, num_examples = test(model, crypto_test_data[crypto], tf.zeros([1, model.num_lstm_units]))
        correct = accuracy * num_examples
        total_examples += num_examples
        total_correct += correct
        p_value = 1 - binom.cdf(correct, num_examples, 0.5)
        print("{:<10}|{:<10}|{:<10}".format(crypto, "{:.2%}".format(accuracy), "{:.4f}".format(p_value)).center(67))

    overall_accuracy = total_correct / total_examples
    overall_p_value = 1 - binom.cdf(total_correct, total_examples, 0.5)
    print("{:<10}|{:<10}|{:<10}".format("-"*10, "-"*10, "-"*10).center(67))
    print("{:<10}|{:<10}|{:<10}".format("Overall", "{:.2%}".format(overall_accuracy), "{:.4f}".format(overall_p_value)).center(67))

if __name__ == '__main__':
    main()
