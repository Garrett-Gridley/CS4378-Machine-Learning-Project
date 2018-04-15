from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, LSTM
import keras
import pandas as pd
import sys
import numpy as np
import pickle
from statistics import stdev
from collections import Counter
from sklearn.preprocessing import LabelBinarizer

df = pd.read_pickle('mbti_1.pickle')
vocab = pd.read_pickle('mbti_1_vocab.pickle')

# one hot the labels
labels = df['type']
encoder = LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)
df['type'] = np.array(encoder.fit_transform(labels))

print(max(df['words_per_post']), '±', stdev(df['words_per_post']), 'words per post')

split_frac = 0.8
split_loc = int(vocab.shape[0]*split_frac)

train_features = vocab[:split_loc]
train_labels = df['type'][:split_loc]
print('features: {}\nlabels: {}'.format(train_features.shape, train_labels.shape))
test_features = vocab[split_loc:]
test_labels = df['type'][split_loc:]

model = Sequential()
model.add(Embedding(train_features.shape[0], 128))
model.add(LSTM(128,dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
			  optimizer='adam',
			  metrics=['accuracy'])

model.fit(train_features, train_labels,
	batch_size=32,
	epochs=1,
	validation_data=(test_features, test_labels))

with open('keras_LSTM.pickle', 'wb') as file:
	pickle.dump(model, file)

score,acc = model.evaluate(test_features, test_labels,
	batch_size=32)


print(score, acc)