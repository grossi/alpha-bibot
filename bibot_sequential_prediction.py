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
	klines5m = pd.DataFrame(client.get_historical_klines(pair, '1h', str(window) + " hours ago UTC"), dtype='float')

	data = []
	data.append(klines5m[[1]].values)

	return data

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

# Plots a graph with the predicted output
def plotPredictions(data):
	# Draw a plot comparing the prediction with the real values
	plt.plot(data, color='b', label = 'Predicted Values')
	plt.xlabel('Days')
	plt.ylabel('Price')
	plt.legend()
	plt.show()

if __name__ == "__main__":
	pair = "BTCUSDT"

	model = load_model('bibotPriceOnly.h5')

	window = model.input_shape[1]

	data_raw = pullData(pair, window)

	data, unnormalize = normalizeValues(data_raw)

	print(str(len(data[0])))

	for i in range(64):
		prediction = model.predict(data[:,-window:,:])
		data = np.append(data,[prediction],axis=1)

	#print("5m " + str(makePrediction(model, data5mNorm, unnormalize5m)))
	print(str(len(data[0])))
	plotPredictions(unnormalizeValues(data, unnormalize))
