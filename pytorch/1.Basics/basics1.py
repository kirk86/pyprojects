import torch as tch
from torch.autograd import Variable
import torch.nn as nn
from torchvision import datasets, transforms, models
import numpy as np

# 1.Basic autograd example            L: 21-36
# 2.Basics autograd example 2         L: 39-77
# 3.loading data from numpy           L: 80-83
# 4.Implementing the input pipeline   L: 86-113
# 5.Input pipeline for custom dataset L: 1116-138
# 6.Using pretrained model            L: 141-155
# 7.Save and load model               L: 158-165

# Autograd example 1
# create tensors
x = Variable(tch.Tensor([1]), requires_grad=True)
w = Variable(tch.Tensor([2]), requires_grad=True)
b = Variable(tch.Tensor([3]), requires_grad=True)
print(x, x.grad, w, w.grad, b, b.grad)

# build a computational graph
y = w * x + b  # 2 * 1 + 3 = 5
print(y)

# compute gradients
y.backward()

# print out the gradients
print(x.grad)
print(w.grad)
print(b.grad)

# Autograd example 2
# create tensors
x = Variable(tch.randn(5, 3))
y = Variable(tch.randn(5, 2))

# build a linear layer
linear = nn.Linear(3, 2)
print("w: ", linear.weight)
print("b: ", linear.bias)
print("parameters: ", linear.parameters())
for i in linear.parameters():
    print i

# build loss and optimizer
criterion = nn.MSELoss()
optimizer = tch.optim.SGD(linear.parameters(), lr=0.01)

# forward propagation
pred = linear(x)

# compute loss
loss = criterion(pred, y)
print("loss prrior to optimization: ", loss.data)
print("loss prrior to optimization: ", loss.data[0])

# backpropagation
loss.backward()

# print out the gradients
print("dL/dw: ", linear.weight.grad)
print("dL/db: ", linear.bias.grad)

# perform 1-step of optimization  gradient
optimizer.step()

# print out the loss after 1-step of optimization
pred = linear(x)
loss = criterion(pred, y)
print("loss after 1-step optimization: ", loss.data)

# optimization at low level
linear.weight.data.sub_(0.01 * linear.weight.grad.data)
linear.bias.data.sub_(0.01 * linear.bias.grad.data)
print("loss after low level optimization : ", loss.data)

# loading data from numpy
a = np.array([[1, 2], [3, 4]])  # numpy array
b = tch.from_numpy(a)  # torch tensor
c = b.numpy()  # numpy array

# imput pipelne
# Download and construct dataset
train_dataset = datasets.CIFAR10(root='./data',
                                 train=True,
                                 transform=transforms.ToTensor(),
                                 download=True)

# select one data pair (read data from disk)
image, label = train_dataset[0]
print(image.size())
print(label)

# data loader (provides queue and thread in a simple way)
train_loader = tch.utils.data.DataLoader(dataset=train_dataset,
                                         batch_size=100,
                                         shuffle=True,
                                         num_workers=2)

# when iteration starts queue and thread start to load dataset from files
data_iter = iter(train_loader)

# mini-batch data and labels
images, labels = data_iter.next()

# usage of data loader is as follows:
for img, lbl in train_loader:
    pass


# Input pipeline for custom dataset
# You should build custom dataset as below
class CusstomDataset(tch.utils.data.Dataset):
    def __init__(self):
        # TODO
        # 1. Initialize file path to list of file names
        pass

    def __len__(self):
        # Change 0 to the total sizeof your dataset
        return 0

# then use torch's pre-built datat loader
custom_dataset = CustomDataset()
train_loader = tch.utils.data.DataLoader(dataset=custom_dataset,
                                         batch_size=100,
                                         shuffle=True,
                                         num_workers=2)

# using pre-trained model
# Download and load pre-trained resnet
resnet = models.resnet18(pretrained=True)

# finetuning only the top layer of the model
for param in resnet.parameters():
    param.requires_grad = False

# replace top layer for finetuning
resnet.fc = nn.Linear(resnet.fc.in_features, 100)  # 100 is for example

# for testing
images = Variable(tch.randn(10, 3, 256, 256))
outputs = resnet(images)
print(outputs.size())  # (10, 100)

# save and load the model
# save the load the entire model
tch.save(resnet, 'model.pkl')
model = tch.load('model.pkl')

# save and load only the model parameters
tch.save(resnet.state_dict(), 'params.pkl')
resnet.load_state_dict(tch.load('params.pkl'))
