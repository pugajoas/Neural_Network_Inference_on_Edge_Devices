import time
import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D
from keras.utils.vis_utils import model_to_dot
import matplotlib.pyplot as plt
from keras.utils import np_utils
import os

parser = argparse.ArgumentParser(description="Recibe los parametros necesarios del numero de epochs y el dispositivo a utilizar")
parser.add_argument('--numero',type=int, default = 5)
parser.add_argument('--dispositivo',type=str, choices = ['cpu','gpu'], default = 'cpu')
args = parser.parse_args()
nepochs = args.numero
dispositivo = args.dispositivo

if dispositivo == 'gpu' and tf.config.list_physical_devices('GPU'):
	dispositivo = '/gpu:0'
else:
	dispositivo = '/cpu:0'

with tf.device(dispositivo):
	IMG_ROWS = 28
	IMG_COLS = 28
	NUM_CLASSES = 10
	TEST_SIZE = 0.2
	RANDOM_STATE = 2018
	#MODELO 
	NO_EPOCHS = nepochs
	BATCH_SIZE = 16

	PATH = "fashionmnist/"

	train_file = PATH+"fashion-mnist_train.csv"
	test_file  = PATH+"fashion-mnist_test.csv"

	train_data = pd.read_csv(train_file)
	test_data = pd.read_csv(test_file)

	def data_preprocessing(raw):
	    out_y = keras.utils.np_utils.to_categorical(raw.label, NUM_CLASSES)
	    num_images = raw.shape[0]
	    x_as_array = raw.values[:,1:]
	    x_shaped_array = x_as_array.reshape(num_images, IMG_ROWS, IMG_COLS, 1)
	    out_x = x_shaped_array / 255
	    return out_x, out_y  
	    
	 # prepare the data
	X, y = data_preprocessing(train_data)
	X_test, y_test = data_preprocessing(test_data)

	# Unir las matrices X y X_test
	X = np.concatenate((X, X_test), axis=0)
	y = np.concatenate((y, y_test), axis=0)

	print("Fashion MNIST -  rows:",X.shape[0]," columns:", X.shape[1:4])

	# Model
	model = Sequential()
	# Add convolution 2D
	model.add(Conv2D(32, kernel_size=(3, 3),
			 activation='relu',
			 kernel_initializer='he_normal',
			 input_shape=(IMG_ROWS, IMG_COLS, 1)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.25))
	model.add(Conv2D(64, 
			 kernel_size=(3, 3), 
			 activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(Dropout(0.4))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.3))
	model.add(Dense(NUM_CLASSES, activation='softmax'))


	model.compile(loss=keras.losses.categorical_crossentropy,
		      optimizer='adam',
		      metrics=['accuracy'])

	model.summary()

	k = 5
	cross_val = KFold(k, shuffle=True, random_state = 1)
	fold_count = 1
	histories = []
	eval_scores = []
	eval_scoreloss = []
	times = []

	for train,test in cross_val.split(X):
	    print("Fold ",fold_count)
	    fold_count = fold_count + 1
	    X_train, y_train = X[train], y[train]
	    X_test, y_test = X[test], y[test]
	    start_time = time.perf_counter()

	    history = model.fit(X_train, y_train,
		      batch_size=BATCH_SIZE,
		      epochs=NO_EPOCHS,
		      verbose=1,
		      )
		      
	    end_time = time.perf_counter()
	    eval_loss, eval_accuracy = model.evaluate(X_test, y_test)
	    histories.append(history)
	    eval_scores.append(eval_accuracy)
	    eval_scoreloss.append(eval_loss)
	    elapsed_time = end_time - start_time
	    times.append(elapsed_time)

	i = 0
	float2 = "{0:.2f}"
	for score in eval_scores:
	    percent = score * 100
	    print("Fold-{}: {}%".format(i+1, float2.format(percent)))
	    i = i + 1
	    
	i = 0
	float2 = "{0:.2f}"
	for score in eval_scoreloss:
	    percent = score * 100
	    print("Fold-{} Loss: {}%".format(i+1, float2.format(percent)))
	    i = i + 1
	    
	    
	i = 0
	for time in times:
	    print("Fold-{} Tiempo: {}".format(i+1, float2.format(time)))
	    i = i + 1
