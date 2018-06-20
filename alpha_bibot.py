from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers import Bidirectional
from keras.models import Sequential
from binance.client import Client
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime

import myBinanceKey

# The client needs an API key from Binance.com
client = Client(myBinanceKey.token, myBinanceKey.key)
 
# Pulls data using the Binance API
def pullData(pair):
	klines = pd.DataFrame(client.get_historical_klines(pair, Client.KLINE_INTERVAL_5MINUTE, "30 days ago UTC"), dtype='float')
	data_raw = klines[[1, 2, 3, 4, 5, 8]].values
	dates_raw = klines[[0]].values
	dates = []
	for date in dates_raw:
		dates.append( datetime.datetime.fromtimestamp( ( date[0]/1e3 ) ) )
	return dates, data_raw

# Pulls data in various time intervals from Binance API
def pullVariousData(pair):
	klines5m = pd.DataFrame(client.get_historical_klines(pair, Client.KLINE_INTERVAL_5MINUTE, "30 days ago UTC"), dtype='float')
	klines15m = pd.DataFrame(client.get_historical_klines(pair, Client.KLINE_INTERVAL_15MINUTE, "90 days ago UTC"), dtype='float')
	klines1h = pd.DataFrame(client.get_historical_klines(pair, Client.KLINE_INTERVAL_1HOUR, "360 days ago UTC"), dtype='float')
	klines8h = pd.DataFrame(client.get_historical_klines(pair, Client.KLINE_INTERVAL_8HOUR, "600 days ago UTC"), dtype='float')
	klines = pd.concat([klines8h, klines1h, klines15m, klines5m])
	data_raw = klines[[1, 2, 3, 4, 5, 8]].values
	dates_raw = klines[[0]].values
	dates = []
	for date in dates_raw:
		dates.append( datetime.datetime.fromtimestamp( ( date[0]/1e3 ) ) )
	return dates, data_raw

# Pulls only price data in various time intervals from Binance API
def pullPriceData(pair):
	klines = pd.DataFrame(client.get_historical_klines(pair, Client.KLINE_INTERVAL_5MINUTE, "300 days ago UTC"), dtype='float')
	dates_raw = klines[[0]].values
	prices = klines[[1]].values
	dates = []
	for date in dates_raw:
		dates.append( datetime.datetime.fromtimestamp( ( date[0]/1e3 ) ) )
	return dates, prices

# Turns a 2D array into a 3D array by turning each row into a 2D array
#   with window_size past values
def addDimension(data, window_size):
	r = []
	for i in range(len(data)-window_size):
		r.append(data[i:i+window_size])
	return r

# Normalizes the 3D data returning a numpy.array
def normalizeValues(data):
	# Creates a numpy
	r = np.array(data)
	# Creates an array of 0's in the same shape as the data
	r0 = np.zeros_like(r)
	# Each value is divided by the first in the sequence and subtracted 1
	# The first value is skipped since it's always 0 
	r0[:,1:,:] = r[:,1:,:] / r[:,0:1,:] - 1
	unwind = r[:,0:1,:]
	return r0, unwind

# Splits data into (ratio) of training data and (1-ratio) of test data
# ratio must be in (0,1)
def splitData(data, ratio):
	data_len = data.shape[0]
	split_point = int(round(data_len * ratio))
	trainingData = data[0:split_point,:,:]
	testData = data[split_point:,:,:]
	return trainingData, testData

def createKerasModel(window_size, input_shape, dropout_value,
					activation_function, loss_function, optimizer):
	# Since the last element of each window will be used as the goal Y we subtract 1
	window_size = window_size - 1

	model = Sequential()

	model.add(Bidirectional(LSTM(window_size, return_sequences=True), input_shape=input_shape))
	model.add(Dropout(dropout_value))

	model.add(Bidirectional(LSTM((window_size*2), return_sequences=True)))
	model.add(Dropout(dropout_value))

	model.add(Bidirectional(LSTM(window_size, return_sequences=False)))
	model.add(Dropout(dropout_value))

	model.add(Dense(units=1))

	model.add(Activation(activation_function))

	model.compile(loss=loss_function, optimizer=optimizer)

	return model

# Plots a graph comparing the test output to a predicted output
def testModel(model, X_input, Y_input, unnormalize):
	# Create the prediction based on X_input
	Y_predicted = model.predict(X_input)

	# Unnormalized the outputs
	Y_real = unnormalizeValues(Y_input, unnormalize)
	Y_predicted_real = unnormalizeValues(Y_predicted, unnormalize)

	# Draw a plot comparing the prediction with the real values
	plt.plot(Y_predicted_real, color='b', label = 'Predicted Values')
	plt.plot(Y_real, color='orange', label = 'Real Values')
	plt.xlabel('Days')
	plt.ylabel('Price')
	plt.legend()
	plt.show()

# Extracts the Y values from the normalized values
def unnormalizeValues(data, unnormalize):
	unnormalize = unnormalize[:,:,0:1].flatten()
	unnormalize = unnormalize[-len(data):]
	data = data.flatten()
	return (data + 1) * unnormalize

if __name__ == "__main__":
	# Trading pair to look up 
	pair = "BTCUSDT"
	# Defining hyper parameters
	window_size = 128
	dropout_value = 0.3
	activation_function = 'linear'
	loss_function = 'mse'
	optimizer = 'adam'

	# Pulls the data from Binance
	dates, dates_raw = pullVariousData(pair)

	# Creates a 3D array so each input turns into a list containing of past inputs
	X_input = addDimension(dates_raw.tolist(), window_size)
	X_input, unnormalize = normalizeValues(X_input)
	np.random.shuffle(X_input)

	# Splits 96% of the data for training and 4% for testing
	X_training, X_testing = splitData(X_input, 0.96)

	# Splits the data into sequence and result
	X_training, Y_training = X_training[:,:-1], X_training[:,-1,0:1]
	X_testing, Y_testing = X_testing[:,:-1], X_testing[:,-1,0:1]

	# Creates and trains the model
	model = createKerasModel(window_size, (window_size-1, X_input.shape[-1]),
		dropout_value, activation_function, loss_function, optimizer )
	print(model.summary())
	model.fit(X_training, Y_training, epochs=2, batch_size=32)

	model.save('bibot2.h5')

	# testModel(model, X_testing, Y_testing, unnormalize)