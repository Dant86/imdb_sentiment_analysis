import tensorflow as tf
from conceptnet5.vectors.query import VectorSpaceWrapper
from wordfreq import simple_tokenize
from sklearn.utils import shuffle
import os
import numpy as np

sentiments = ["neg", "pos"]
data_sets = ["IMDBData/train/", "IMDBData/test/"]
vec_dirname = "../conceptnet5/data/precomputed/vectors/mini.h5"
vocab = []
wembs = []
labels = []
docs = []
wrapper = VectorSpaceWrapper(vec_dirname)

print("Getting vocab and word embeddings...")
for data_set in data_sets:
	for sentiment in sentiments:
		dirname = data_set + sentiment + "/"
		for fname in os.listdir(dirname):
			with open(dirname + fname) as doc:
				unprocessed = doc.read()
				tokenized = simple_tokenize(unprocessed, "en")
				docs.append(tokenized)
				if sentiment == "pos":
					labels.append([0, 1])
				else:
					labels.append([1, 0])
				for word in tokenized:
					if word not in vocab:
						print(word)
						vocab.append(word)
						wembs.append(wrapper.text_to_vector("en", word))

print("Processing data...")
docs, labels = shuffle(docs, labels)
split_train = 0.8
train_docs = docs[:int(split_train*len(docs))]
train_labels = np.array(labels[:int(split_train*len(labels))]).T
test_docs = docs[int(split_train*len(docs)):]
test_labels = np.array(labels[int(split_train*len(labels)):]).T

def encode(doc):
	return sum([wembs[vocab.index(word)] for word in doc])

train_vecs = np.array([encode(doc) for doc in train_docs]).T
test_vecs = np.array([encode(doc) for doc in test_docs]).T

print(np.shape(train_vecs[:,1]))


