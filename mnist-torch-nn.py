import time
import torch
import torchvision
import torch.utils.data
import numpy as np
import tensorflow as tf
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

start = time.localtime(time.time())

n_epochs = 20
batch_size_train = 256
batch_size_test = 100
learning_rate = 0.0005
momentum = 0.5
log_interval = 10
n_of_filters = 64

device = torch.device("cuda")


train_loader = torch.utils.data.DataLoader(
	torchvision.datasets.MNIST('/files/', train=True, download=True,
							 transform=torchvision.transforms.Compose([
							   torchvision.transforms.ToTensor(),
							   torchvision.transforms.Normalize(
								 (0.1307,), (0.3081,))
							 ])),
	batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
	torchvision.datasets.MNIST('/files/', train=False, download=True,
							 transform=torchvision.transforms.Compose([
							   torchvision.transforms.ToTensor(),
							   torchvision.transforms.Normalize(
								 (0.1307,), (0.3081,))
							 ])),
	batch_size=batch_size_test, shuffle=True)


input_shape = (batch_size_train,28,28)



class Mnist_Net(nn.Module):
	def __init__(self, input_shape):
		super(Mnist_Net, self).__init__()
		self.conv_pipe = nn.Sequential(
				nn.Conv2d(in_channels=1, out_channels=n_of_filters,
						  kernel_size=4, stride=1, padding=1),
				nn.ReLU(),
				nn.Dropout2d(p=0.3), 
				nn.Conv2d(in_channels=n_of_filters, out_channels=n_of_filters*2,
						  kernel_size=4, stride=1, padding=1),
				nn.ReLU(),
				nn.Dropout2d(p=0.3),
				nn.MaxPool2d(kernel_size = 2),
				nn.ReLU(),
				nn.Dropout2d(p=0.2),
				nn.Conv2d(in_channels=n_of_filters*2, out_channels=n_of_filters*4,
						  kernel_size=4, stride=1, padding=1),
				nn.ReLU(),
				nn.Dropout2d(p=0.3),
				nn.MaxPool2d(kernel_size = 2),
				nn.ReLU(),
				nn.Dropout2d(p=0.2)
				)


		self.dense_pipe =  nn.Sequential(
				
				nn.Linear(n_of_filters*6*6*4, 128),
				nn.Sigmoid(),
				nn.Dropout2d(p=0.1),
				nn.Linear(128, 10),
				nn.Softmax()
			)


	def forward(self, x):	
		conv =  self.conv_pipe(x)
		#print(np.shape(conv))
		conv = conv.view(-1, int(n_of_filters*6*6*4))
		return self.dense_pipe(conv)



net = Mnist_Net(input_shape).to(device)


def target_converter(target):

	target_tensor = []

	for number in target:
		vector = np.zeros(10)
		vector[number] = 1
		target_tensor.append(vector)


	return torch.FloatTensor(target_tensor)	





def train(net, train_loader, test_loader, n_of_epochs):

	loss_objective = nn.CrossEntropyLoss()
	optimizer =  optim.Adam(params=net.parameters(), lr=learning_rate, betas=(momentum, 0.999))

	for epoch in range(n_of_epochs):	

		for step, (batch, target) in enumerate(train_loader):
			optimizer.zero_grad()
			batch = batch.to(device)
			target = target.to(device)
			model_output = net.forward(batch)
		
			loss = loss_objective(model_output, target.long())
			loss.backward()
			optimizer.step()

		test(net, test_loader)


def test(net, test_loader):

	correct = 0

	for step, (batch, target) in enumerate(test_loader):
		batch = batch.to(device)
		model_output = net(batch)
		pred = model_output.to("cpu").data.max(1, keepdim=True)[1]
		correct += pred.eq(target.data.view_as(pred)).sum()

	print("\nAccuracy:  ", float(correct)/10000.0)  	

train(net, train_loader, test_loader, n_epochs)


print("\n")

end = time.localtime(time.time())
start_in_sec = start[3]*3600 + start[4]*60 + start[5]
end_in_sec = end[3]*3600 + end[4]*60 + end[5]


all_time_min = int((end_in_sec-start_in_sec)/60)
all_time_sec = (end_in_sec-start_in_sec)%60
if all_time_min < 10:
	if all_time_sec < 10:
		print('0%s:0%s' % (all_time_min, all_time_sec ))
	else:
		print('0%s:%s' % (all_time_min, all_time_sec ))	
else:
	if all_time_sec < 10:
		print('%s:0%s' % (all_time_min, all_time_sec ))
	else:
		print('%s:%s' % (all_time_min, all_time_sec ))	