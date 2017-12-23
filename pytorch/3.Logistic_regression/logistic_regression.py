import torch as tch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.autograd import Variable

# Hyper parameters
input_size = 784
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# MNIST dataset (images and labels)
train_dataset = datasets.MNIST(root='./data',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='./data',
                              train=False,
                              transform=transforms.ToTensor())

# Dataset loader (Input Pipeline)
train_loader = tch.utils.data.DataLoader(dataset=train_dataset,
                                         batch_size=batch_size,
                                         shuffle=True)

test_loader = tch.utils.data.DataLoader(dataset=test_dataset,
                                        batch_size=batch_size,
                                        shuffle=True)

# Model
class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)


    def forward(self, x):
        out = self.linear(x)
        return out


model = LogisticRegression(input_size, num_classes)

# Set loss + optimizer
# Softmax is internally computed
# set parameters to be updated
criterion = nn.CrossEntropyLoss()
optimizer = tch.optim.SGD(model.parameters(), lr=learning_rate)

# training model
for epoch in xrange(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.view(-1, 28 * 28))
        labels = Variable(labels)

        # forward + backward + optimize
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print("Epoch: [{}/{}], Step: [{}/{}], Loss: {:4f}"
                  .format(epoch + 1, num_epochs, i + 1,
                          len(train_dataset)//batch_size, loss.data[0]))

# Test the model
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images.view(-1, 28*28))
    outputs = model(images)
    _, predicted = tch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print("Accuracy of the model on the 10000 test images: {} %%"
      .format(100 * correct / total))

# save the model
# tch.save(model.state_dict(), 'model.pkl')
