import torch.nn as nn
from masked_convolutional_layer import MaskedConv2d
from utils import *


class ConditionalMaskedConv2d(MaskedConv2d):
    """
    Class extending nn.Conv2d to use masks and condition on class
    """

    def __init__(self, mask_type, num_classes, in_channels, out_channels, kernel_size, padding=0):
        super(ConditionalMaskedConv2d, self).__init__(mask_type, in_channels, out_channels, kernel_size,
                                                      padding=padding)
        self.V = nn.Parameter(torch.randn(out_channels, num_classes))

    def forward(self, x, class_condition):
        conv_output = super().forward(x)
        s = conv_output.shape
        conv_output = conv_output.view(s[0], s[1], s[2] * s[3]) + (self.V @ class_condition.T).T.unsqueeze(-1)
        return conv_output.reshape(s)


class ConditionalPixelCNN(nn.Module):
    """
    Class conditional PixelCNN-class
    """

    def __init__(self, in_channels, conv_filters, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            # A 7x7 A-type convolution
            ConditionalMaskedConv2d('A', num_classes, in_channels=in_channels, out_channels=conv_filters, kernel_size=7,
                                    padding=3),
            nn.ReLU(),
            # 5 7x7 B-type convolutions
            ConditionalMaskedConv2d('B', num_classes, in_channels=conv_filters, out_channels=conv_filters,
                                    kernel_size=7, padding=3),
            nn.ReLU(),
            ConditionalMaskedConv2d('B', num_classes, in_channels=conv_filters, out_channels=conv_filters,
                                    kernel_size=7, padding=3),
            nn.ReLU(),
            ConditionalMaskedConv2d('B', num_classes, in_channels=conv_filters, out_channels=conv_filters,
                                    kernel_size=7, padding=3),
            nn.ReLU(),
            ConditionalMaskedConv2d('B', num_classes, in_channels=conv_filters, out_channels=conv_filters,
                                    kernel_size=7, padding=3),
            nn.ReLU(),
            ConditionalMaskedConv2d('B', num_classes, in_channels=conv_filters, out_channels=conv_filters,
                                    kernel_size=7, padding=3),
            nn.ReLU(),
            # 2 1x1 B-type convolutions
            ConditionalMaskedConv2d('B', num_classes, in_channels=conv_filters, out_channels=conv_filters,
                                    kernel_size=1),
            nn.ReLU(),
            ConditionalMaskedConv2d('B', num_classes, in_channels=conv_filters, out_channels=in_channels,
                                    kernel_size=1),
            nn.Sigmoid()).cuda()

    def forward(self, x, class_condition):
        out = x
        for layer in self.net:
            if isinstance(layer, ConditionalMaskedConv2d):
                out = layer(out, class_condition)
            else:
                out = layer(out)
        return out


def train_conditional_pixelcnn(train_data, train_labels, test_data, test_labels, image_shape, n_classes, dset_id):
    """
    train_data: A (n_train, H, W, 1) numpy array of binary images with values in {0, 1}
    train_labels: A (n_train,) numpy array of class labels
    test_data: A (n_test, H, W, 1) numpy array of binary images with values in {0, 1}
    test_labels: A (n_test,) numpy array of class labels
    image_shape: (H, W), height and width
    n_classes: number of classes (4 or 10)
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
            used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (100, H, C, 1) of samples with values in {0, 1}
    where an even number of images of each class are sampled with 100 total
    """

    def one_hot_labels(labels):
        labels_oh = np.zeros((labels.size, n_classes))
        labels_oh[np.arange(labels.size), labels] = 1
        return torch.tensor(labels_oh).float().cuda()

    train_labels = one_hot_labels(train_labels)
    test_labels = one_hot_labels(test_labels)

    def normalize(x):
        return 2 * (x - 0.5)

    def cross_entropy_loss(batch, output):
        return torch.nn.functional.binary_cross_entropy(output, batch)

    def get_batched_loss(dataset, labels, model):
        test_loss = []
        for batch, label in zip(torch.split(dataset, 128), torch.split(labels, 128)):
            out = model(normalize(batch), label)
            loss = cross_entropy_loss(batch, out)
            test_loss.append(loss.item())

        return np.mean(np.array(test_loss))

    train_data = torch.from_numpy(np.transpose(train_data, [0, 3, 1, 2])).float().cuda()
    test_data = torch.from_numpy(np.transpose(test_data, [0, 3, 1, 2])).float().cuda()

    dataset_params = {
        'batch_size': 128,
        'shuffle': False
    }

    n_epochs = 20 if dset_id == 1 else 10
    lr = 1e-3
    no_channels, convolution_filters = 1, 64

    cpixelcnn = ConditionalPixelCNN(no_channels, convolution_filters, num_classes=n_classes).cuda()
    train_loader = torch.utils.data.DataLoader(train_data, **dataset_params)
    train_label_loader = torch.utils.data.DataLoader(train_labels, **dataset_params)
    optimizer = torch.optim.Adam(cpixelcnn.parameters(), lr=lr)

    train_losses = []
    test_losses = [get_batched_loss(test_data, test_labels, cpixelcnn)]

    for epoch in range(n_epochs):
        for batch_x, batch_y in zip(train_loader, train_label_loader):
            optimizer.zero_grad()
            output = cpixelcnn(normalize(batch_x), batch_y)
            loss = cross_entropy_loss(batch_x, output)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        test_loss = get_batched_loss(test_data, test_labels, cpixelcnn)
        test_losses.append(test_loss)
        print(f"[{100*(epoch+1)/n_epochs:.2f}%] Epoch {epoch + 1}")

    torch.cuda.empty_cache()

    H, W = image_shape
    samples = torch.zeros(size=(100, 1, H, W)).cuda()
    sample_classes = one_hot_labels(np.sort(np.array([np.arange(n_classes)] * (100 // n_classes)).flatten()))

    cpixelcnn.eval()
    with torch.no_grad():
        for i in range(H):
            for j in range(W):
                out = cpixelcnn(normalize(samples), sample_classes)
                torch.bernoulli(out[:, :, i, j], out=samples[:, :, i, j])

    return np.array(train_losses), np.array(test_losses), np.transpose(samples.detach().cpu().numpy(), [0, 2, 3, 1])


if __name__ == '__main__':
    # Dataset 1 (Shapes)
    show_results_conditional_1c(1, train_conditional_pixelcnn)

    # Dataset 2 (MNIST)
    show_results_conditional_1c(2, train_conditional_pixelcnn)
