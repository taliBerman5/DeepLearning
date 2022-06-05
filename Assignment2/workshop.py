# conda create --name DPL_assignment2 python=3.7
# conda activate DPL_assignment2
# conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=10.2 -c pytorch
# torch.cuda.is_available()    check if i have cuda
# torch.cuda.device_count()    check how much device i have


import torch.optim as optim
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F


# set seeds for numpy, torch
def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# data
classes = ("plane", "", "", "", "", "", "", "", "", "track")
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((.5, .5, .5), (.5, .5, .5))])  # gets data to [-1,1]

trainset = torchvision.datasets.CIFAR10(root="../data/", train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)
testset = torchvision.datasets.CIFAR10(root="../data/", train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)

dataiter = iter(trainloader)
images, labels = dataiter.next()


# network
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)  # in channels, out channels, kernel size
        self.pool = torch.nn.MaxPool2d(2, 2)  # size,stride
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        # two convolutional layers, a bunch of ff layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# training
set_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Model().to(device)
opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = torch.nn.CrossEntropyLoss()

sum(p.numel() for p in model.parameters() if
    p.requires_grad)  # prints the amount of paramaters that needs grad - sums for all layers

# iterate over epochs
for epoch in range(2):
    total_loss = 0.0

    # iterate over the dataset
    for i, data in enumerate(trainloader):
        inputs, labels = data

        opt.zero_grad()

        # forward pass
        outputs = model(inputs.to(device))
        loss = criterion(outputs, labels)

        # backward pass
        loss.backward()
        opt.step()

        # print stats
        total_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d,%5d] loss: %.3f' % (epoch + 1, i + 1, total_loss / 2000))
            total_loss = 0

PATH = './cifar_model.pth'
torch.save(model.state_dict(), PATH)  # saves only the dict and not the whole model

# testing

model.eval()  # in order to run on the test data
with torch.no_grad():  # below this line no grads are calculated
    total, correct = 0, 0
    # iterate over the test data
    for data in testloader:
        images, labels = data

        # fwd pass
        outputs = model(images)

        # get predicted label
        _, pred = torch.max(outputs.data,
                            1)  # maximum over the first dimension (because we want the maximum per sample)
        # max, argmax = torch.max(outputs.data, 1)

        # comapre to true label
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    print("Accuracy of the network on the test set %d %%" % (100 * correct / total))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, pred = torch.max(outputs, 1)

        c = (pred == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
    for i in range(10):
        print("Accuracy of %5s : %2d %%" % (classes[i], 100 * class_correct[i] / class_total[i]))
