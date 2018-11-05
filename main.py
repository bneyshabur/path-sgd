import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import copy
import argparse
from path_sgd import PathSGD
import time
from torchvision import transforms, datasets

# train the model for one epoch on the given set
def train(args, model, state_dict, device, train_loader, criterion, optimizer, path_optimizer, epoch):
    sum_loss, sum_correct = 0, 0

    # switch to train mode
    model.train()

    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device).view(data.size(0),-1), target.to(device)

        # compute the output
        output = model(data)

        # compute the classification error and loss
        loss = criterion(output, target)
        pred = output.max(1)[1]
        sum_correct += pred.eq(target).sum().item()
        sum_loss += len(data) * loss.item()

        # compute the gradient and do an SGD step
        optimizer.zero_grad()
        loss.backward()

        if args.optimizer == 'path-sgd':
            path_optimizer.update_grad()

        optimizer.step()
    return 1 - (sum_correct / len(train_loader.dataset)), sum_loss / len(train_loader.dataset)


# evaluate the model on the given set
def validate(args, model, device, val_loader, criterion):
    sum_loss, sum_correct = 0, 0

    # switch to evaluation mode
    model.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(val_loader):
            data, target = data.to(device).view(data.size(0), -1), target.to(device)

            # compute the output
            output = model(data)

            # computer the classification error and margin
            pred = output.max(1)[1]
            sum_correct += pred.eq(target).sum().item()
            sum_loss += len(data) * criterion(output, target).item()
    return 1 - (sum_correct / len(val_loader.dataset)), sum_loss / len(val_loader.dataset)


# Load and Preprocess data.
# Loading: If the dataset is not in the given directory, it will be downloaded.
# Preprocessing: This includes normalizing each channel and data augmentation by random cropping and horizontal flipping
def load_data(split, dataset_name, datadir, nchannels):

    if dataset_name == 'MNIST':
        normalize = transforms.Normalize(mean=[0.131], std=[0.289])
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    tr_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(), normalize])
    val_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize])

    get_dataset = getattr(datasets, dataset_name)
    if dataset_name == 'SVHN':
        if split == 'train':
            dataset = get_dataset(root=datadir, split='train', download=True, transform=tr_transform)
        else:
            dataset = get_dataset(root=datadir, split='test', download=True, transform=val_transform)
    else:
        if split == 'train':
            dataset = get_dataset(root=datadir, train=True, download=True, transform=tr_transform)
        else:
            dataset = get_dataset(root=datadir, train=False, download=True, transform=val_transform)

    return dataset


# This function trains a fully connected neural net with two hidden layer using SGD or Path-SGD
def main():

    # settings
    parser = argparse.ArgumentParser(description='Training a fully connected NN with one hidden layer')
    parser.add_argument('--no-cuda', default=False, action='store_true',
                        help='disables CUDA training')
    parser.add_argument('--datadir', default='datasets', type=str,
                        help='path to the directory that contains the datasets (default: datasets)')
    parser.add_argument('--dataset', default='CIFAR10', type=str,
                        help='name of the dataset (options: MNIST | CIFAR10 | CIFAR100 | SVHN, default: CIFAR10)')
    parser.add_argument('--nunits', default=4000, type=int,
                        help='number of hidden units (default: 4000)')
    parser.add_argument('--optimizer', default='sgd', type=str,
                        help='name of the optimizer (options: sgd | path-sgd, default: sgd)')
    parser.add_argument('--epochs', default=1000, type=int,
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--stopcond', default=0.01, type=float,
                        help='stopping condtion based on the cross-entropy loss (default: 0.01)')
    parser.add_argument('--batchsize', default=64, type=int,
                        help='input batch size (default: 64)')
    parser.add_argument('--learningrate', default=0.01, type=float,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum (default: 0.9)')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    nchannels, nclasses = 3, 10
    if args.dataset == 'MNIST': nchannels = 1
    if args.dataset == 'CIFAR100': nclasses = 100
    input_dim = 32 * 32 * nchannels
    # create an initial model
    model = nn.Sequential(nn.Linear(input_dim, args.nunits), nn.ReLU(),
                            nn.Linear(args.nunits, args.nunits), nn.ReLU(), nn.Linear(args.nunits, nclasses)).to(device)

    state_dict = model.state_dict()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), args.learningrate, momentum=args.momentum)

    path_optimizer = None
    if args.optimizer == 'path-sgd':
        path_optimizer = PathSGD(model, input_dim)

    # loading data
    train_dataset = load_data('train', args.dataset, args.datadir, nchannels)
    val_dataset = load_data('val', args.dataset, args.datadir, nchannels)

    train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, **kwargs)
    val_loader = DataLoader(val_dataset, batch_size=args.batchsize, shuffle=False, **kwargs)

    # training the model
    for epoch in range(0, args.epochs):
        start_time = time.time()
        # train for one epoch
        tr_err, tr_loss = train(args, model, state_dict, device, train_loader, criterion, optimizer, path_optimizer, epoch)
        val_err, val_loss = validate(args, model, device, val_loader, criterion)

        elapsed_time = time.time()-start_time
        print(f'Epoch: {epoch + 1}/{args.epochs}\t Elapsed time: {elapsed_time:.3f}\t training loss: {tr_loss:.3f}\t ',
                f'Training error: {tr_err:.3f}\t Validation error: {val_err:.3f}')

        # stop training if the cross-entropy loss is less than the stopping condition
        if tr_loss < args.stopcond: break

if __name__ == '__main__':
    main()
