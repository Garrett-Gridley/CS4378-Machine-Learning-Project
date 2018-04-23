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

model = Sequential()
model.add(Embedding(train_features.shape[0], 128))
model.add(LSTM(128))
model.add(Dense(16, activation='relu'))
model.compile(loss='categorical_crossentropy',
			  optimizer='adam',
			  metrics=['categorical_accuracy'])

model.fit(train_features, train_labels,
	batch_size=batch_size,
	epochs=5,
	validation_data=(test_features, test_labels))

predictions = model.predict(test_features)
classes = predictions.argmax(axis=-1)
print(classes)

# score,acc = model.evaluate(test_features, test_labels,
# 	batch_size=batch_size)

# print(score, acc)

# ENFP = "Recently, I learned about the mbti types of my family members, and my sister is definitely an ENFP. Which explained so many things she had to go through, aka depression. But the thing is that when she was going through it, I don’t remember that me and my family were aware of it in the first place and I don’t think that we were of a great support to her at that time... and I’m just thankful that she is healthy and alright now. Sooooo....I want to ask you guys how to help an ENFP out of their depression??? What worked for you guys and how did you deal with it??"

# vectorizer = CountVectorizer(max_features=1000)
# person = vectorizer.fit_transform([ENFP])
# print(model.predict(person))

model.save('keras_LSTM.h5')
