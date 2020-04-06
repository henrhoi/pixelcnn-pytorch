import torch.nn as nn
from masked_convolutional_layer import MaskedConv2d
from utils import *


class PixelCNN(nn.Module):
    """
    PixelCNN-class
    """

    def __init__(self, in_channels, conv_filters):
        super().__init__()
        self.net = nn.Sequential(
            # A 7x7 A-type convolution
            MaskedConv2d('A', in_channels=in_channels, out_channels=conv_filters, kernel_size=7, padding=3),
            nn.ReLU(),
            # 5 7x7 B-type convolutions
            MaskedConv2d('B', in_channels=conv_filters, out_channels=conv_filters, kernel_size=7, padding=3),
            nn.ReLU(),
            MaskedConv2d('B', in_channels=conv_filters, out_channels=conv_filters, kernel_size=7, padding=3),
            nn.ReLU(),
            MaskedConv2d('B', in_channels=conv_filters, out_channels=conv_filters, kernel_size=7, padding=3),
            nn.ReLU(),
            MaskedConv2d('B', in_channels=conv_filters, out_channels=conv_filters, kernel_size=7, padding=3),
            nn.ReLU(),
            MaskedConv2d('B', in_channels=conv_filters, out_channels=conv_filters, kernel_size=7, padding=3),
            nn.ReLU(),
            # 2 1x1 B-type convolutions
            MaskedConv2d('B', in_channels=conv_filters, out_channels=conv_filters, kernel_size=1),
            nn.ReLU(),
            MaskedConv2d('B', in_channels=conv_filters, out_channels=in_channels, kernel_size=1),
            nn.Sigmoid()).cuda()

    def forward(self, x):
        return self.net(x)


def train_pixelcnn(train_data, test_data, image_shape, dset_id):
    """
    train_data: A (n_train, H, W, 1) uint8 numpy array of binary images with values in {0, 1}
    test_data: A (n_test, H, W, 1) uint8 numpy array of binary images with values in {0, 1}
    image_shape: (H, W), height and width of the image
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
            used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (100, H, W, 1) of samples with values in {0, 1}
    """

    def normalize(x):
        return 2 * (x - 0.5)

    def cross_entropy_loss(batch, output):
        return torch.nn.functional.binary_cross_entropy(output, batch)

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

    n_epochs = 15 if dset_id == 1 else 10
    lr = 1e-3
    no_channels, convolution_filters = 1, 64

    pixelcnn = PixelCNN(no_channels, convolution_filters).cuda()
    train_loader = torch.utils.data.DataLoader(train_data, **dataset_params)
    optimizer = torch.optim.Adam(pixelcnn.parameters(), lr=lr)

    train_losses = []
    test_losses = [get_batched_loss(test_data, pixelcnn)]

    for epoch in range(n_epochs):
        for batch_x in train_loader:
            optimizer.zero_grad()
            output = pixelcnn(normalize(batch_x))
            loss = cross_entropy_loss(batch_x, output)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        test_loss = get_batched_loss(test_data, pixelcnn)
        test_losses.append(test_loss)
        print(f"[{100*(epoch+1)/n_epochs:.2f}%] Epoch {epoch + 1}")

    torch.cuda.empty_cache()

    H, W = image_shape
    samples = torch.zeros(size=(100, 1, H, W)).cuda()

    pixelcnn.eval()
    with torch.no_grad():
        for i in range(H):
            for j in range(W):
                out = pixelcnn(normalize(samples))
                torch.bernoulli(out[:, :, i, j], out=samples[:, :, i, j])

    return np.array(train_losses), np.array(test_losses), np.transpose(samples.detach().cpu().numpy(), [0, 2, 3, 1])


if __name__ == '__main__':
    # Dataset 1 (Shapes)
    show_results_1c(1, train_pixelcnn)

    # Dataset 2 (MNIST)
    show_results_1c(2, train_pixelcnn)
