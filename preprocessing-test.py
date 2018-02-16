# -*- coding: utf-8 -*-
import csv
from sklearn.preprocessing import LabelEncoder, StandardScaler
import re
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import keras.optimizers
from keras.utils import plot_model

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sets import Set
import numpy as np

def get_x_train():
	train_file = open('train_set_x.csv', 'rb')
	reader = csv.reader(train_file, delimiter = ',')

	rows = []
	for line in reader:
		rows.append(line)
	del(rows[0])

	for row in rows:
		del(row[0])

	rows = map(lambda x: x[0].lower().decode('utf-8'), rows)
	rows = map(lambda x: re.sub(r'(\s)http\w+','',x), rows)
	return rows

def get_x_test():
	test_file = open('test_set_x.csv', 'rb')
	reader = csv.reader(test_file, delimiter = ',')

	entries = []
	for a,b in reader:
		entries.append(b)
	del(entries[0])

	rows = []
	for string in entries:
		s1 = ''.join(string.split())
		rows.append(s1)

	rows = map(lambda x: x.lower().decode('utf-8'), rows)
	rows = map(lambda x: re.sub(r'(\s)http\w+','',x), rows)
	return rows



#takes as input a SCALED X, and ONE-HOT encoded Y
def output_train_test_files(X,Y,x_t):
	#seed to randomize
	#seed = 216

	#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=seed)

	#save our train/test data 
	#np.savez_compressed('train_test_data.npz',X_train=X_train,Y_train=Y_train,X_test=X_test,Y_test=Y_test)
	np.savez_compressed('train_test_data-allfeatures-tfidf-nohttp.npz',X_train=X,Y_train=Y,X_test=x_t)
	print("Data has been saved.")

	#	returns one-hot encoded y data
def preprocess_y():
	train_file = open('train_set_y.csv', 'rb')
	reader = csv.reader(train_file, delimiter = ',')

	rows = []
	for a,b in reader:
		rows.append(b)
	del(rows[0])

	#traansform string to int
	rows = map(lambda x: int(x), rows)

	#one-hot encoding with keras
	y = keras.utils.to_categorical(rows, num_classes=5)

	return y


vectorizer = TfidfVectorizer(analyzer='char', lowercase = False, max_features=200)

#print(get_x_train())
#print(rows)
X_train = vectorizer.fit_transform(get_x_train())
X_test = vectorizer.transform(get_x_test())

Y_train = preprocess_y()



output_file = open('nn-results-allfeatures-tfidf-nohttp.csv', 'wb')

model = Sequential()
model.add(Dense(1000,input_dim=200,kernel_initializer="glorot_uniform",activation="sigmoid"))
model.add(Dropout(0.5))
model.add(Dense(600,kernel_initializer="glorot_uniform",activation="sigmoid"))
model.add(Dropout(0.5))
model.add(Dense(200,kernel_initializer="glorot_uniform",activation="sigmoid"))
model.add(Dropout(0.5))

#we have 5 categories
categories = 5

model.add(Dense(categories,kernel_initializer="glorot_uniform",activation="softmax"))
model_optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#model_optimizer = 'rmsprop'

model.compile(loss='categorical_crossentropy',
			  optimizer=model_optimizer,
			  metrics=['accuracy'])

history = model.fit(X_train,Y_train,
		  epochs=12,
		  validation_split=0.10,
		  batch_size=32,
		  verbose=2,
		  shuffle=True)

results = model.predict_classes(X_test, batch_size=64, verbose=0)

output = []
for i in range(len(results)):
	output.append(('{},{}'.format(i,results[i])))

output_file.write('Id,Category\n')
for line in output:
	output_file.write(line + '\n')
output_file.close()

#print(vectorizer.get_feature_names())
#print(rows[5], x[5])
