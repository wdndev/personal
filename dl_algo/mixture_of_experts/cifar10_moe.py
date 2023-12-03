

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from moe import MoE

def train(trainloader, epochs, model, loss_fn, optimizer):
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            inputs = inputs.view(inputs.shape[0], -1)
            outputs, aux_loss = model(inputs)
            # calculate prediction loss
            loss = loss_fn(outputs, labels)
            # combine losses
            total_loss = loss + aux_loss

            total_loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
        torch.save(model.state_dict(), "model/weights/model-{}.pth".format(epoch))
    print('Finished Training')
    return model


def eval(testloader, model):
    correct = 0
    total = 0
    model.eval()
    # model returns the prediction and the loss that encourages all experts to have equal importance and load
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs, aux_loss = model(images.view(images.shape[0], -1))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))


def load_dataset(batch_size=16, cifar10_path="./data"):
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainloader, testloader, classes

if __name__ == '__main__':
    # arguments
    input_size = 3072
    num_classes = 10
    num_experts = 10
    hidden_size = 256
    k = 4

    batch_size = 16
    

    # determine device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # instantiate the MoE layer
    model = MoE(input_size, num_classes, num_experts, hidden_size, k=k, noisy_gating=True)
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optim = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    trainloader, testloader, _ = load_dataset(batch_size=batch_size)

    # train
    model = train(trainloader, 50, model, loss_fn, optim)

    # model_weight_path = "model/weights/model-9.pth"
    # model.load_state_dict(torch.load(model_weight_path, map_location=device))
    # evaluate
    eval(testloader, model)
