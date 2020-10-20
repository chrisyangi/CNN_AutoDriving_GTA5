import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from ResNet import resnet34
import torch.optim as optim
from random import shuffle
import pandas as pd
import os
#data_transform = transforms.Compose([
#                                 transforms.ToTensor(),
#                                 transforms.Normalize([0.406], [0.224])])
WIDTH = 160
HEIGHT = 120
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

train_data = np.load('training_data.npy')
shuffle(train_data)
train = train_data[:-2000]
test = train_data[-2000:]
#train -= np.mean(train,axis = 0)
#train /= np.std(train,axis = 0)
#print(train[:2])
#test -= np.mean(test,axis = 0)
#test /= np.std(test,axis = 0)

X = np.array([i[0] for i in train]).reshape(-1,1,WIDTH,HEIGHT)
Y = [i[1] for i in train]
testX = np.array([i[0] for i in test]).reshape(-1,1,WIDTH,HEIGHT)
testY = [i[1] for i in test]
X = torch.from_numpy(X)
X = X.type(torch.FloatTensor)
Y = torch.from_numpy(np.array(Y))
Y = Y.type(torch.LongTensor)
testX = torch.from_numpy(testX)
testY = torch.from_numpy(np.array(testY))
testX = testX.type(torch.FloatTensor)
testY = testY.type(torch.LongTensor)
net = resnet34()

inchannel = net.fc.in_features
net.fc = nn.Linear(inchannel,9)
model_weight_path = './ResNet34.pth'
net.load_state_dict(torch.load(model_weight_path),strict = False)
net.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)
batch_size = 128
best_acc = 0.0
save_path = './ResNet34.pth'
lenX = len(X) - len(X) % batch_size
lentestX = len(testX) - len(testX) % batch_size
print("train start.....")
for epoch in range(2):
	net.train()
	running_loss = 0.0
	i = 0
	while i < lenX:
		img = X[i:i+batch_size]
		img = img.to(device)
		label = Y[i:i+batch_size]
		label = torch.max(label,1)[1]
		label = label.to(device)
		optimizer.zero_grad()
		logits = net(img)
		loss = loss_function(logits,label)
		loss.backward()
		optimizer.step()
		running_loss += loss.item()
		i = i + batch_size
		rate = i / lenX
		a = "*" * int(rate * 50)
		b = "." * int((1 - rate) * 50)
		print("\rtrain loss: {:^3.0f}%[{}->{}]{:.4f}".format(int(rate*100), a, b, loss), end="")
	print()
	net.eval()
	acc = 0.0
	j = 0
	with torch.no_grad():
		while j < lentestX:
			img = testX[j:j+batch_size]
			img = img.to(device)
			label = testY[j:j+batch_size]
			label = torch.max(label,1)[1]
			label = label.to(device)
			output = net(img)
			predict_y = torch.max(output,dim = 1)[1]
			acc += (predict_y == label).sum().item()
			j = j + batch_size
			val_acc = acc / lentestX
			if val_acc > best_acc:
				best_acc = val_acc
				torch.save(net.state_dict(),save_path)
		print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' % (epoch + 1, running_loss / (lenX / batch_size), val_acc))

print('Finished Training')
