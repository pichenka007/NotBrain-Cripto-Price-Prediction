from tvDatafeed import TvDatafeed, Interval
import numpy as np
import os
import keras
from keras import layers
import tensorflow as tf
from tensorflow import data as tf_data
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint
import time
os.environ["KERAS_BACKEND"] = "tensorflow"

ckpt = 0


load = True # True/False


import math

def sig(x):
	return 1 / (1 + math.exp(-x))

history = 96

tv = TvDatafeed()

"""
Interval.in_1_minute
Interval.in_3_minute
Interval.in_5_minute
Interval.in_15_minute
Interval.in_30_minute
Interval.in_45_minute
Interval.in_1_hour
Interval.in_2_hour
Interval.in_3_hour
Interval.in_4_hour
Interval.in_daily
Interval.in_weekly
Interval.in_monthly
"""

df = tv.get_hist(symbol="TONUSDT", exchange="OKX", interval=Interval.in_15_minute, n_bars=96*30)

print(df.keys())

dopen = df["open"].to_numpy()
dhigh = df["high"].to_numpy()
dlow = df["low"].to_numpy()
dclose = df["close"].to_numpy()
dvolume = df["volume"].to_numpy()
dvolume = dvolume/np.mean(dvolume)

data = list(zip(dopen, dhigh, dlow, dclose, dvolume))

train_x = []
train_y = []
for i in range(len(dopen)-history-1):
	train_x.append([])
	for q in range(history):
		train_x[i].append([*data[i+q]])#, *data[i+q]
	train_y.append([*data[i+history]])

print("len train:", len(train_x))

early_stop = EarlyStopping(monitor='val_acc', min_delta=0.001,
						   patience=5, verbose=1, mode='auto')

chkpt = ModelCheckpoint("ckpt/ckpt-{epoch}.keras",
						monitor='val_loss',
						verbose=1,
						mode='auto')

if load:
	model = keras.models.load_model("model.keras")
else:
	model = keras.Sequential([
		layers.LSTM(512, input_shape=(history, 5), return_sequences=True),
		layers.Dropout(.2),
		layers.LSTM(256, return_sequences=True),
		layers.LSTM(256, return_sequences=False),
		layers.Dropout(.2),
		layers.Dense(5, activation="linear")],
	)

	model.compile(
		optimizer="adam",
		loss="mse",
		metrics=["accuracy"],
	)

while True:
	try:
		print("enter ctrl+c to stop fit and start plot predicts")
		model.fit(
			np.array(train_x),
			np.array(train_y),
			validation_split=0.05,
			epochs=10**11,
			shuffle=True,
			callbacks=[early_stop, chkpt],
		)
		break
	except:
		try:
			print("enter ctrl+c to save model and exit")

			pred = []
			for i in range(history-1):
				pred.append([0, 0, 0, 0, 0])

			predd = model.predict(train_x)
			for p in predd:
				pred.append(list(p))
			prognoz = int(input("len predicts?: "))
			for i in range(prognoz):
				delta = []

				for q in range(history):
					delta.append(pred[len(pred)-history-1+q])

				pred.append(list(model.predict(np.array([delta], dtype=np.float32))[0]))
			for i in range(len(pred)):
				pred[i] = pred[i][0]

			plt.plot(dopen)
			plt.plot(pred)
			plt.show()
		except:
			break


model.save('model.keras')
model.save_weights("model_w.h5")