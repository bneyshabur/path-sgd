# Path-SGD: Path-Normalized Optimization in Deep Neural Networks

This repository contains the code to train neural nets uising Path-SGD optimization method:

**[Path-SGD: Path-Normalized Optimization in Deep Neural Networks](http://arxiv.org/abs/1506.02617)**

[Behnam Neyshabur](https://cs.nyu.edu/~behnam/), [Ruslan Salakhutdinov](http://www.cs.cmu.edu/~rsalakhu/), [Nathan Srebro](http://www.ttic.edu/srebro)

## Usage
1. Install *Python 3.6* and *PyTorch 0.4.1*.
2. Clone the repository:
   ```
   git clone https://github.com/bneyshabur/path-sgd.git
   ```
3. The following command trains a fully connected feedforward network with two hidden layer, each of which with 4000 hidden units on *CIFAR10* dataset, using Path-SGD:
   ```
   python main.py --dataset CIFAR10 --optimizer path-sgd
   ```
## Main Inputs Arguments
* `--no-cuda`: disables cuda training
* `--datadir`: path to the directory that contains the datasets (default: datasets)
* `--dataset`: name of the dataset (options: MNIST | CIFAR10 | CIFAR100 | SVHN, default: CIFAR10). If the dataset is not in the desired directory, it will be downloaded.
* `--nunits`: number of hidden units (default: 4000)
* `--optimizer`: name of the optimizer (options: sgd | path-sgd, default: sgd).
