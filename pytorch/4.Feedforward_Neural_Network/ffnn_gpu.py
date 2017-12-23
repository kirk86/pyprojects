import torch as tch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.autograd import Variable

# Hyper-parameters

input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 1e-3

# MNIST dataset
train_dataset = datasets.MNIST(root='./data',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='./data',
                              train=True,
                              transform=transforms.ToTensor(),
                              download=True)

# Data loader (Input pipeline)
train_loader = tch.utils.data.DataLoader(dataset=train_dataset,
                                         batch_size=batch_size,
                                         shuffle=True)

test_loader = tch.utils.data.DataLoader(dataset=test_dataset,
                                        batch_size=batch_size,
                                        shuffle=False)


# Feedforward Neural Network (1 hidden layer)
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


net = Net(input_size, hidden_size, num_classes)
net.cuda()

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = tch.optim.Adam(net.parameters(), lr=learning_rate)

for epoch in xrange(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Convert torch tensor to variable
        images = Variable(images.view(-1, 28 * 28)).cuda()
        labels = Variable(labels).cuda()

        # forward + backward + optimize
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print("Epoch [{}/{}], Step [{}/{}], Loss: {:4f}"
                  .format(epoch + 1, num_epochs, i + 1,
                          len(train_dataset)//batch_size, loss.data[0]))

# Test the model
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images.view(-1, 28 * 28)).cuda()
    outputs = net(images).cuda()
    _, predicted = tch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print("Accuracy of the network on the 10000 test images: {}%"
      .format(100 * correct / total))

# save the model
# tch.save(net.state_dict(), 'model.pkl')
