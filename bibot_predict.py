from keras.models import Sequential
from keras.models import load_model
from binance.client import Client
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime

import myBinanceKey

# The client needs an API key from Binance.com
client = Client(myBinanceKey.token, myBinanceKey.key)

def pullData(pair, window):
	klines5m = pd.DataFrame(client.get_historical_klines(pair, '5m', str(window*5) + " minutes ago UTC"), dtype='float')
	klines15m = pd.DataFrame(client.get_historical_klines(pair, '15m', str(window*15) + " minutes ago UTC"), dtype='float')
	klines1h = pd.DataFrame(client.get_historical_klines(pair, '1h', str(window) + " hours ago UTC"), dtype='float')
	klines8h = pd.DataFrame(client.get_historical_klines(pair, '8h', str(window*8) + " hours ago UTC"), dtype='float')

	data5m = []
	data5m.append(klines5m[[1, 2, 3, 4, 5, 8]].values)
	data15m = []
	data15m.append(klines15m[[1, 2, 3, 4, 5, 8]].values)
	data1h = []
	data1h.append(klines1h[[1, 2, 3, 4, 5, 8]].values)
	data8h = []
	data8h.append(klines8h[[1, 2, 3, 4, 5, 8]].values)

	return data5m, data15m, data1h, data8h

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

# Returns a prediction of the next price
def makePrediction(model, X_input, unnormalize):
	# Create the prediction based on X_input
	Y_predicted = model.predict(X_input)

	return unnormalizeValues(Y_predicted, unnormalize)

# Extracts the Y values from the normalized values
def unnormalizeValues(data, unnormalize):
	unnormalize = unnormalize[:,:,0:1].flatten()
	unnormalize = unnormalize[-len(data):]
	data = data.flatten()
	return (data + 1) * unnormalize

if __name__ == "__main__":
	pair = "BTCUSDT"

	model = load_model('bibot.h5')

	window = model.input_shape[1]

	data5m, data15m, data1h, data8h = pullData(pair, window)

	data5mNorm, unnormalize5m = normalizeValues(data5m)
	data15mNorm, unnormalize15m = normalizeValues(data15m)
	data1hNorm, unnormalize1h = normalizeValues(data1h)
	data8hNorm, unnormalize8h = normalizeValues(data8h)

	print("5m " + str(makePrediction(model, data5mNorm, unnormalize5m)))
	print("15m " + str(makePrediction(model, data15mNorm, unnormalize15m)))
	print("1h " + str(makePrediction(model, data1hNorm, unnormalize1h)))
	print("8h " + str(makePrediction(model, data8hNorm, unnormalize8h)))
