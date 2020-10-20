import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle

train_data = np.load('training_data.npy')

df = pd.DataFrame(train_data)
print(df.head())
print(Counter(df[1].apply(str)))
forwards = []
lefts = []
rights = []
backs = []
forwardAndLefts = []
forwardAndRights = []
backAndLefts = []
backAndRights = []
noKeys = []
for data in train_data:
	img = data[0]
	choice = data[1]
	if(choice == [1,0,0,0,0,0,0,0,0]):
		forwards.append([img,choice])
	elif(choice == [0,1,0,0,0,0,0,0,0]):
		lefts.append([img,choice])
	elif(choice == [0,0,1,0,0,0,0,0,0]):
		rights.append([img,choice])
	elif(choice == [0,0,0,1,0,0,0,0,0]):
		backs.append([img,choice])
	elif(choice == [0,0,0,0,1,0,0,0,0]):
		forwardAndLefts.append([img,choice])
	elif(choice == [0,0,0,0,0,1,0,0,0]):
		forwardAndRights.append([img,choice])
	elif(choice == [0,0,0,0,0,0,1,0,0]):
		backAndLefts.append([img,choice])
	elif(choice == [0,0,0,0,0,0,0,1,0]):
		backAndRights.append([img,choice])
	elif(choice == [0,0,0,0,0,0,0,0,1]):
		noKeys.append([img,choice])
	else:
		print("No key to match")
		
forwards = forwards[:len(lefts)][:len(rights)]
lefts = lefts[:len(forwards)]
rights = rights[:len(forwards)]
backs = backs[:len(forwards)]
forwardAndLefts = forwardAndLefts[:len(forwards)]
forwardAndRights = forwardAndRights[:len(forwards)]
backAndLefts = backAndLefts[:len(forwards)]
backAndRights = backAndRights[:len(forwards)]
noKeys = noKeys[:len(forwards)]
final_data = forwards + lefts + rights + backs + forwardAndLefts + forwardAndRights + backAndLefts + backAndRights + noKeys;
shuffle(final_data)
print(len(final_data))

dff = pd.DataFrame(final_data)
print(dff.head())
print(Counter(dff[1].apply(str)))

