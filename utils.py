import pickle
from os.path import join
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import torch


def load_pickled_data(fname, include_labels=False):
    with open(fname, 'rb') as f:
        data = pickle.load(f)

    train_data, test_data = data['train'], data['test']
    if 'mnist.pkl' in fname or 'shapes.pkl' in fname:
        # Binarize MNIST and shapes dataset
        train_data = (train_data > 127.5).astype('uint8')
        test_data = (test_data > 127.5).astype('uint8')
    if 'celeb.pkl' in fname:
        train_data = train_data[:, :, :, [2, 1, 0]]
        test_data = test_data[:, :, :, [2, 1, 0]]
    if include_labels:
        return train_data, test_data, data['train_labels'], data['test_labels']
    return train_data, test_data


def show_samples(samples, nrow=10, title='Samples'):
    samples = (torch.FloatTensor(samples) / 255).permute(0, 3, 1, 2)
    grid_img = make_grid(samples, nrow=nrow)
    plt.figure()
    plt.title(title)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis('off')
    plt.show()


def show_training_plot(train_losses, test_losses, title):
    plt.figure()
    n_epochs = len(test_losses) - 1
    x_train = np.linspace(0, n_epochs, len(train_losses))
    x_test = np.arange(n_epochs + 1)

    plt.plot(x_train, train_losses, label='train loss')
    plt.plot(x_test, test_losses, label='test loss')
    plt.legend()
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('NLL')
    plt.show()


def show_results_1c(dset_type, fn):
    """
    Show results for datasets with one channels
    """
    data_dir = "data"
    if dset_type == 1:
        train_data, test_data = load_pickled_data(join(data_dir, 'shapes.pkl'))
        img_shape = (20, 20)
    elif dset_type == 2:
        train_data, test_data = load_pickled_data(join(data_dir, 'mnist.pkl'))
        img_shape = (28, 28)
    else:
        raise Exception()

    train_losses, test_losses, samples = fn(train_data, test_data, img_shape, dset_type)
    samples = samples.astype('float32') * 255

    print(f'Final Test Loss: {test_losses[-1]:.4f}')
    show_training_plot(train_losses, test_losses, f'Dataset {dset_type} Train Plot')
    show_samples(samples)


def show_results_3c(dset_type, fn):
    """
    Show results for datasets with three channels
    """
    data_dir = "data"
    if dset_type == 1:
        train_data, test_data = load_pickled_data(join(data_dir, 'shapes_colored.pkl'))
        img_shape = (20, 20, 3)
    elif dset_type == 2:
        train_data, test_data = load_pickled_data(join(data_dir, 'mnist_colored.pkl'))
        img_shape = (28, 28, 3)
    else:
        raise Exception()

    train_losses, test_losses, samples = fn(train_data, test_data, img_shape, dset_type)
    samples = samples.astype('float32') / 3 * 255

    print(f'Final Test Loss: {test_losses[-1]:.4f}')
    show_training_plot(train_losses, test_losses, f'Dataset {dset_type} Train Plot')
    show_samples(samples)


def show_results_conditional_1c(dset_type, fn):
    data_dir = "data"
    if dset_type == 1:
        train_data, test_data, train_labels, test_labels = load_pickled_data(join(data_dir, 'shapes.pkl'),
                                                                             include_labels=True)
        img_shape, n_classes = (20, 20), 4
    elif dset_type == 2:
        train_data, test_data, train_labels, test_labels = load_pickled_data(join(data_dir, 'mnist.pkl'),
                                                                             include_labels=True)
        img_shape, n_classes = (28, 28), 10
    else:
        raise Exception('Invalid dset type:', dset_type)

    train_losses, test_losses, samples = fn(train_data, train_labels, test_data, test_labels, img_shape, n_classes,
                                              dset_type)
    samples = samples.astype('float32') * 255

    print(f'Final Test Loss: {test_losses[-1]:.4f}')
    show_training_plot(train_losses, test_losses, f'Q3(d) Dataset {dset_type} Train Plot')
    show_samples(samples)
