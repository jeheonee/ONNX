import argparse

import torch
import torch.nn as nn
import torch.onnx as onnx
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_resnet18(args):
    """ Get a trained ResNet-18 model by ImageNet """

    model = models.resnet18(num_classes=args.num_classes).cuda()
    return model


def get_cifar10(train=True, args=None):
    """ Get a CIFAR-10 dataloader """

    # Image pre-processing
    transform = transforms.Compose([transforms.ToTensor()])
    
    # CIFAR-10 dataset
    cifar = datasets.CIFAR10(root='./data',
                             train=train,
                             transform=transform,
                             download=True)
    
    # CIFAR-10 dataloader
    cifar_loader = DataLoader(dataset=cifar,
                              batch_size=args.batch,
                              shuffle=True)
    
    return cifar_loader


def train(model, dataloader, args):
    """ Train a model using dataloader """

    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr)

    for i in range(args.epochs):

        num_data = 0
        total_acc = 0.0
        total_loss = 0.0

        for images, labels in dataloader:
            images = images.cuda()
            labels = labels.cuda()

            # Predict labels and compute loss
            preds = model(images)
            loss = criterion(preds, labels)

            # Optimize the models
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update total loss and total accuracy
            num_data += len(images)
            total_acc += (preds.max(1)[1] == labels).sum().item()
            total_loss += loss.item()

        total_acc = total_acc / num_data
        total_loss = total_loss / len(dataloader)

        print('Epoch [{:4}/{:4}] Loss: {:.5f}, Acc: {:3.3f}%'.format(i+1, args.epochs, total_loss, total_acc * 100))

    return model


def convert_onnx(model, args):
    """ Convert the PyTorch model to the ONNX file """

    dummy_data = torch.empty(1, 3, *args.img_shape, dtype = torch.float32)
    model.cpu()

    onnx.export(model, dummy_data, args.filename, input_names=['input'], output_names=['output'])


if __name__ == '__main__':
    # Get arguments from the user
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--img_shape', type=list, default=[32, 32])
    parser.add_argument('--filename', type=str, default='resnet18_cifar10.onnx')
    args = parser.parse_args()

    # Get a ResNet
    model = get_resnet18(args)

    # Get a train CIFAR-10 dataloader
    train_loader = get_cifar10(args=args)

    # Training
    train(model, train_loader, args)

    # Save the ONNX file
    convert_onnx(model, args)
