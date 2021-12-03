import tensorflow as tf
import numpy as np
import csv
from functools import reduce

def get_data(data_file):
    """
    Read and parse the train line by line, then breaks up the line into price, market cap and total volume.

    :param data_file: Path to the training file.
    :return: list of price(percent change), market cap(log scale) and total volume(percent change) and name
    """
    # TODO: load and concatenate training data from training file.
    with open(data_file, 'r') as csvfile:
        prices = []
        market_cap = []
        volume = []
        for row in csv.reader(csvfile, delimiter=','):
            prices.append(row[1])
            market_cap.append(row[2])
            volume.append(row[3])
    data_file = data_file[7:]
    name = data_file.split("-",1)[0]
    prices = prices[1:]
    market_cap = market_cap[2:]
    volume = volume[1:]
    prices = list(map(float, prices))
    percent_change = []
    for i in range(len(prices) - 1):
        percent_change.append((prices[i + 1] - prices[i])/prices[i])
    market_cap = list(map(float, market_cap))
    volume = list(map(float, volume))
    volume_change = []
    for i in range(len(volume) - 1):
        volume_change.append((volume[i + 1] - volume[i])/volume[i])
    percent_change = np.asarray(percent_change)
    market_cap = np.asarray(market_cap)
    log_market_cap = np.log(market_cap)
    volume_change = np.asarray(volume_change)
    return np.row_stack([normalize(percent_change), normalize(log_market_cap), normalize(volume_change)]), name


def normalize(array):
    return (array - np.mean(array)) / np.std(array)

# price_change, log_market_cap, volume_change = get_data("./data/ada-usd-max.csv")
# print(price_change)

    