import torch as tch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np


# Hyper parameters
input_size = 1
output_size = 1
num_epochs = 60
learning_rate = 0.001

# Toy dataset
trX = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                [9.779], [6.182], [7.59], [2.167], [7.042],
                [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

trY = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                [3.366], [2.596], [2.53], [1.221], [2.827],
                [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)


class LinearRegression(nn.Module):
    """Linear regression class"""

    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.linear(x)
        return out


model = LinearRegression(input_size, output_size)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = tch.optim.SGD(model.parameters(), lr=learning_rate)

# Train the model
for epoch in xrange(num_epochs):
    # convert numpy array to torch
    inputs = Variable(tch.from_numpy(trX))
    targets = Variable(tch.from_numpy(trY))

    # forward + backward + optimize
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    if (epoch) % 5 == 0:
        # print("Epoch [%d/%d], Loss: %.4f"
        #       % (epoch + 1, num_epochs, loss.data[0]))
        print("Epoch [{}/{}], Loss: {:4f}"
              .format(epoch+1, num_epochs, loss.data[0]))

# plot actual data and regression outcome
predictions = model(inputs)
plt.plot(trX, trY, 'ro', label='Original data')
plt.plot(trX, predictions.data.numpy(), label='Fitted line')
plt.legend()
plt.show()

# save the model
# tch.save(model.state_dict(), 'model.pkl')
