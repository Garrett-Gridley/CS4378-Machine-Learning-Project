from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, LSTM, TimeDistributed
import keras
import pandas as pd
import sys
import numpy as np
import h5py

import pickle
from statistics import stdev
from sklearn.preprocessing import OneHotEncoder

import time

df = pd.read_pickle('mbti_1.pickle')
vocab = pd.read_pickle('mbti_1_vocab.pickle')
word_map = pd.read_pickle('mbti_1_word_map.pickle')

# convert posts to integers
# word_ints = []
# for row in df['posts']:
# 	ints = []
# 	for word in row.split():
# 		if word in word_map:
# 			ints.append(word_map[word])
# 	word_ints.append(ints)

# word_ints = np.array(word_ints)

# one hot the labels
labels = df['type']
now = time.time()
encode = { t:i for i,t in enumerate(set(df['type']))}
decode = { i:t for t,i in encode.items() }

def onehot(x):
	hot = np.zeros(16)
	hot[encode[x]] = 1
	return hot

onehots = np.array([onehot(x) for x in df['type']])
# onehots = df.as_matrix(columns=['onehots'])

print(onehots.shape)
# print(now-time.time())
# encoder = OneHotEncoder()
# df['type'] = np.array(encoder.fit(labels))

# print(max(df['words_per_post']), '±', stdev(df['words_per_post']), 'words per post')

split_frac = 0.8
split_loc = int(vocab.shape[0]*split_frac)

train_features = vocab[:split_loc]
train_labels = onehots[:split_loc]
print('features: {}\nlabels: {}'.format(train_features.shape, train_labels.shape))
test_features = vocab[split_loc:]
test_labels = onehots[split_loc:]

batch_size = 128



if len(sys.argv) > 1 and sys.argv[1] == 'train':

	model = Sequential()
	model.add(Embedding(train_features.shape[0], 128))
	model.add(LSTM(128, dropout=0.5, recurrent_dropout=0.5))
	model.add(Dense(16, activation='relu'))
	model.compile(loss='categorical_crossentropy',
				  optimizer='adam',
				  metrics=['accuracy'])

	model.fit(train_features, train_labels,
		batch_size=batch_size,
		epochs=1,
		validation_data=(test_features, test_labels))

	model.save('keras_LSTM.h5')
if len(sys.argv) > 1 and sys.argv[1] == 'loss':
	from keras.models import load_model
	import matplotlib.pyplot as plt
	model = load_model('keras_LSTM.h5')

	print(model.loss)
	# print(model.__dict__.keys())
	# plt.plot(model.train_history_['loss'])
	# plt.plot(model.train_history_['val_loss'])
	# plt.show()

if len(sys.argv) > 1 and sys.argv[1] == 'confuse':
	from keras.models import load_model 
	from sklearn.metrics import confusion_matrix
	from numpy import argmax
	import matplotlib.pyplot as plt

	model = load_model('keras_LSTM.h5')


	actual = df['type'][split_loc:]
	preds = model.predict(test_features, batch_size=batch_size)
	classes = [decode[i] for i in preds.argmax(axis=-1)]
	
	# print(actual, classes)
	cm = confusion_matrix(actual, classes)

	plt.figure()
	plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
	plt.colorbar()
	plt.show()

else:
	from keras.models import load_model
	model = load_model('keras_LSTM.h5')
	score, acc = model.evaluate(test_features, test_labels, batch_size=batch_size)
	print(score, acc)
