# -*- coding: utf-8 -*-
import csv
from sklearn.preprocessing import LabelEncoder, StandardScaler
import re

from sklearn.model_selection import train_test_split
from sets import Set
import numpy as np

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB



#define our complete alphabet to count our letters
def get_alphabet():
	en = 'abcdefghijklmnopqrstuvwxyz'
	special_chars =' !?¿¡'
	german = 'äöüß'
	french = 'àâæçéèêêîïôœùûüÿ'
	spanish = 'áéíóúüñ'
	slovak = 'áäčďdzdžéíĺľňóôŕšťúýž'
	polish = 'ąćęłńóśźż'
	empty = ''

	full_alphabet = en + special_chars + german + french + spanish + slovak + polish + empty
	#print(full_alphabet)
	#remove any duplicates by converting to set, and then back to a list
	full_alphabet = list(set(list(full_alphabet.decode('utf-8'))))

	full_alphabet = sorted(full_alphabet);
	#print(len(full_alphabet))
	return full_alphabet

#simple bag-of-letters counting of chars
def count_chars(text,alphabet):
	alphabet_counts = []
	for letter in alphabet:
		count = text.count(letter)
		alphabet_counts.append(count)
	return alphabet_counts

#	returns a list of sentences, each consisting of a list of characters
def preprocess_x():
	train_file = open('train_set_x.csv', 'rb')
	reader = csv.reader(train_file, delimiter = ',')

	rows = []
	for line in reader:
		rows.append(line)
	del(rows[0])

	for row in rows:
		del(row[0])

	#The following line does a lot.
	#we decode each row into an unicode string
	#each entry becomes a list of characters
	rows = map(lambda x: x[0].lower().decode('utf-8'), rows)

	
	#print(rows);

	#cleanup emojis, random error symbols
	#get data in the form:
	# 'aace' -> [2,0,1,0,1]
	valid_alphabet = get_alphabet()

	#for each row we will count the chars and write out to a csv 
	x = []
	for i in range(len(rows)):
		bagged = count_chars(rows[i], valid_alphabet)
		x.append(bagged)

	standard_scaler = StandardScaler().fit(x)
	x = standard_scaler.transform(x)

	#print(x)
	return x;
		



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
	#y = keras.utils.to_categorical(rows, num_classes=5)

	return rows



def preprocess_testdata_x():
	file = open('test_set_x.csv', 'rb')
	reader = csv.reader(file, delimiter = ',')

	entries = []
	for a,b in reader:
		entries.append(b)
	del(entries[0])

	rows = []
	for string in entries:
		s1 = string.lower().decode('utf-8')
		s1 = ''.join(s1.split())
		rows.append(s1)

	
	x = []
	for i in range(len(rows)):
		bagged = count_chars(rows[i], get_alphabet())
		x.append(bagged)


	#n = 48
	#print(rows[n])
	#print(get_alphabet())
	#print(x[n])
	#print(x_scaled[n])


	standard_scaler = StandardScaler().fit(x)
	x = standard_scaler.transform(x)

	#print(x)
	return x;



model = MultinomialNB()
model.fit(preprocess_x(), preprocess_y())

results =  model.predict(preprocess_testdata_x())

output = []
for i in range(len(results)):
	output.append(('{},{}'.format(i,results[i])))

output_file = open('nb-results-scaled.csv', 'wb')

output_file.write('Id,Category\n')
for line in output:
	output_file.write(line + '\n')
output_file.close()

