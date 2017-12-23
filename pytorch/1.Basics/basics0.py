import torch
import torchvision
import numpy as np

# Simple tensor example

# random normal
x = torch.randn(5, 3)
print(x)


# build a single layer
linear = torch.nn.Linear(3, 2)

# sess weight and bias
print(linear.weight)
print(linear.bias)

# forward propagation
y = linear(torch.autograd.Variable(x))
print(y)

# convert numpy array to torch tensors
a = np.array([[1, 2], [3, 4]])
b = torch.from_numpy(a)
print(b)

# data transformation and preprocessing
transform = torchvision.transforms.Compose([torchvision.transforms.Scale(4),
                                            torchvision.transforms.RandomHorizontalFlip(),
                                            torchvision.transforms.RandomCrop(32),
                                            torchvision.transforms.ToTensor()])

# defining a dataset
train = torchvision.datasets.CIFAR10(root='./data/',
                                     train=True,
                                     transform=transform,
                                     download=True)

# pick and print an item from the dataset
img, label = train[0]
print(img.size())
print(label)

# data loader
train_loader = torch.utils.data.DataLoader(dataset=train,
                                           batch_size=100,
                                           shuffle=True,
                                           num_workers=2)
data_iter = iter(train_loader)
# mini-batch imgs and labels
imgs, labels = data_iter.next()

for imgs, labels in train_loader:
    # training code goes here
    pass


# custom dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        """ Custom dataset """
       pass

   def __getitem__(self, index):
       # this function should return one data for a given index
       pass

   def __len__(self):
       pass

custom_dataset = CustomDataset()
torch.utils.data.DataLoader(dataset=custom_dataset,
                            batch_size=100,
                            shuffle=True,
                            num_workers=2)

# using pre-trained model
# download and use pre-trained model
resnet = torchvision.models.resnet18(pretrained=True)
# finetuning ==> removing top layer
sub_model = torch.nn.Sequential(*list(resnet.children()[:-1]))

# for test
images = torch.autograd(Variable(torch.randn(10, 3, 256, 256)))
print(resnet(images).size())
print(sub_model(images).size())

# save and load model
torch.save(sub_model, 'model.pkl')
model = torch.load('model.pkl')
