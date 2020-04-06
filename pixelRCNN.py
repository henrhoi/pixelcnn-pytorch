import torch.nn as nn
from masked_convolutional_layer import MaskedConv2d
from utils import *


class ResidualMaskedConv2d(nn.Module):
    """
    Residual Links between MaskedConv2d-layers
    As described in Figure 5 in "Pixel Recurrent Neural Networks" by Aaron van den Oord et. al.
    """

    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            MaskedConv2d('B', in_dim, in_dim // 2, kernel_size=1, padding=0),
            nn.ReLU(),
            MaskedConv2d('B', in_dim // 2, in_dim // 2, kernel_size=7, padding=3),
            nn.ReLU(),
            MaskedConv2d('B', in_dim // 2, in_dim, kernel_size=1, padding=0),
            nn.ReLU())

    def forward(self, x):
        return self.net(x) + x


class PixelRCNN(nn.Module):
    """
    Pixel R-CNN-class that assumes color channels as independent using residual blocks
    from "Pixel Recurrent Neural Networks" by Aaron van den Oord et. al.
    """

    def __init__(self, in_channels, out_channels, conv_filters):
        super().__init__()
        self.net = nn.Sequential(
            # A 7x7 A-type convolution with batch norm
            MaskedConv2d('A', in_channels, conv_filters, kernel_size=7, padding=3),
            nn.BatchNorm2d(conv_filters),
            nn.ReLU(),
            # 8 Residual B-type convolutons with batch norms
            ResidualMaskedConv2d(conv_filters),
            nn.BatchNorm2d(conv_filters),
            nn.ReLU(),
            ResidualMaskedConv2d(conv_filters),
            nn.BatchNorm2d(conv_filters),
            nn.ReLU(),
            ResidualMaskedConv2d(conv_filters),
            nn.BatchNorm2d(conv_filters),
            nn.ReLU(),
            ResidualMaskedConv2d(conv_filters),
            nn.BatchNorm2d(conv_filters),
            nn.ReLU(),
            ResidualMaskedConv2d(conv_filters),
            nn.BatchNorm2d(conv_filters),
            nn.ReLU(),
            ResidualMaskedConv2d(conv_filters),
            nn.BatchNorm2d(conv_filters),
            nn.ReLU(),
            ResidualMaskedConv2d(conv_filters),
            nn.BatchNorm2d(conv_filters),
            nn.ReLU(),
            ResidualMaskedConv2d(conv_filters),
            nn.BatchNorm2d(conv_filters),
            nn.ReLU(),
            MaskedConv2d('B', conv_filters, out_channels, kernel_size=1)).cuda()

    def forward(self, x):
        return self.net(x)


def train_pixelrcnn(train_data, test_data, image_shape, dset_id):
    """
    train_data: A (n_train, H, W, C) uint8 numpy array of color images with values in {0, 1, 2, 3}
    test_data: A (n_test, H, W, C) uint8 numpy array of color images with values in {0, 1, 2, 3}
    image_shape: (H, W, C), height, width, and # of channels of the image
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
            used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (100, H, C, W) of samples with values in {0, 1, 2, 3}
    """
    H, W, C = image_shape
    bits_per_channel = 4

    def normalize(x):
        """
        Values in [0, 3] normalizing to [-1, 1]
        """
        return (x - 1.5) / 1.5

    def get_proba(output):
        return torch.nn.functional.softmax(output.reshape(output.shape[0], bits_per_channel, C, H, W), dim=1)

    def cross_entropy_loss(batch, output):
        per_bit_output = output.reshape(batch.shape[0], bits_per_channel, C, H, W)
        return torch.nn.CrossEntropyLoss()(per_bit_output, batch.long())

    def get_batched_loss(dataset, model):
        test_loss = []
        for batch in torch.split(dataset, 128):
            out = model(normalize(batch))
            loss = cross_entropy_loss(batch, out)
            test_loss.append(loss.item())

        return np.mean(np.array(test_loss))

    train_data = torch.from_numpy(np.transpose(train_data, [0, 3, 1, 2])).float().cuda()
    test_data = torch.from_numpy(np.transpose(test_data, [0, 3, 1, 2])).float().cuda()

    dataset_params = {
        'batch_size': 128,
        'shuffle': True
    }

    n_epochs = 20
    lr = 1e-3
    no_channels, out_channels, convolution_filters = C, C * bits_per_channel, 120

    pixelrcnn = PixelRCNN(no_channels, out_channels, convolution_filters).cuda()
    train_loader = torch.utils.data.DataLoader(train_data, **dataset_params)
    optimizer = torch.optim.Adam(pixelrcnn.parameters(), lr=lr)

    train_losses = []
    test_losses = [get_batched_loss(test_data, pixelrcnn)]

    for epoch in range(n_epochs):
        for batch_x in train_loader:
            optimizer.zero_grad()
            output = pixelrcnn(normalize(batch_x))
            loss = cross_entropy_loss(batch_x, output)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        test_loss = get_batched_loss(test_data, pixelrcnn)
        test_losses.append(test_loss)
        print(f"[{100*(epoch+1)/n_epochs:.2f}%] Epoch {epoch + 1}")

    torch.cuda.empty_cache()

    H, W, C = image_shape
    samples = torch.zeros(size=(100, C, H, W)).cuda()

    pixelrcnn.eval()
    with torch.no_grad():
        for i in range(H):
            for j in range(W):
                for c in range(C):
                    out = pixelrcnn(normalize(samples))
                    proba = get_proba(out)
                    samples[:, c, i, j] = torch.multinomial(proba[:, :, c, i, j], 1).squeeze().float()

    return np.array(train_losses), np.array(test_losses), np.transpose(samples.detach().cpu().numpy(), [0, 2, 3, 1])


if __name__ == '__main__':
    # Dataset 1 (Colorized Shapes)
    show_results_3c(1, train_pixelrcnn)

    # Dataset 2 (Colorized MNIST)
    show_results_3c(2, train_pixelrcnn)