import csv
import pickle 
import re
import pandas as pd
import string

from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter

FILE_PATH = "mbti_1.csv"


url_pattern = r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))'''
data = pd.read_csv(FILE_PATH)

def clean(line):
	line = line.lower()
	line = line.replace('|||', '')
	line = re.sub(url_pattern, '', line, flags=re.MULTILINE)
	line = line.translate({ord(x):'' for x in string.punctuation})
	line = re.sub(r'\s+', ' ', line).strip()

	return line

data['posts'] = data['posts'].apply(clean)

# data['words_per_post'] = data['posts'].apply(lambda x: len(x.split(' ')))
# data['chars_per_post'] = data['posts'].apply(lambda x: len(x))

vectorizer = CountVectorizer(max_features=1000)
vocab = vectorizer.fit_transform(data['posts'])

with open('mbti_1_vocab.pickle', 'wb') as file:
	pickle.dump(vocab, file)

with open('mbti_1_word_map.pickle', 'wb') as file:
	pickle.dump(vectorizer.vocabulary_, file)

# with open('mbti_1.pickle', 'wb') as file:
# 	pickle.dump(data, file)
