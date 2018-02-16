import csv
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np

output_file = open('knn-results.csv', 'wb')

#load our training and testing data
train_test_data = np.load('train_test_data-full.npz')

X_train = train_test_data['X_train']
print ("X_train: ",X_train.shape)

Y_train = train_test_data['Y_train']
print ("Y_train: ",Y_train.shape)

X_test = train_test_data['X_test']
print ("X_test: ",X_test.shape)


model = KNeighborsClassifier(n_neighbors = 5)
model.fit(X_train, Y_train)

results =  model.predict(X_test)

output = []
for i in range(len(results)):
	output.append(('{},{}'.format(i,results[i])))

output_file.write('Id,Category\n')
for line in output:
	output_file.write(line + '\n')
output_file.close()

