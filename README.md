# Pixel CNN
Various PixelCNN implementations using PyTorch, based on [Pixel Recurrent Neural Networks](https://arxiv.org/abs/1601.06759) by van den Oord et. al.


## Models

**PixelCNN:**

Implementation of a simple PixelCNN architecture to model binary MNIST and shape images using the following architecture:

* A 7x7 masked type A convolution
* 5 7x7 masked type B convolutions
* 2 1x1 masked type B convolutions
* Appropriate ReLU nonlinearities in-between
* 64 convolutional filters

*Illustration of context:*

![](https://i.imgur.com/mq1WGaw.png)

**PixelRCNN assuming Independent Color Channels (PixelRCNN):**

A PixelCNN implementation that supports RGB color channels (or augment your existing implementation).
This PixelCNN uses residual masked convolution layers and assumes color channels as independent
More formally, we model the following parameterized distribution:

<img src="https://render.githubusercontent.com/render/math?math=p_\theta(x) = \prod_{i=1}^{HW}\prod_{c=1}^C p_\theta(x_i^c | x_{ < i})">


The following architecture were used:

* A 7×7 masked type A convolution
* 8 residual blocks with masked type B convolutions
* Appropriate ReLU nonlinearities and Batch Normalization in-between
* 128 convolutional filters

**PixelCNN assuming Dependent Color Channels (Autoregressive PixelRCNN):**

A PixelCNN implementation that models **dependent** color channels. Formally, we model the parameterized distribution

<img src="https://render.githubusercontent.com/render/math?math=p_\theta(x) = \prod_{i=1}^{HW}\prod_{c=1}^C p_\theta(x_i^c | x_i^{ < c}, x_{ < i})">.

Masking schemes are changed for the center pixel. 
The filters are split into 3 groups, only allowing each group to see the groups before (or including the current group, for type B masks) to maintain the autoregressive property.

**Conditional PixelCNN:**

A **class-conditional** PixelCNN implementation on binary MNIST. Condition on a class label by adding a conditional bias in each convolutional layer. More precisely, in the <img src="https://render.githubusercontent.com/render/math?math=\ell">th convolutional layer, compute: 

<img src="https://render.githubusercontent.com/render/math?math=W_\ell * x %2B b_\ell %2B V_\ell y">,

where <img src="https://render.githubusercontent.com/render/math?math=W_\ell * x %2B b_\ell"> is a masked convolution (as in previous parts), <img src="https://render.githubusercontent.com/render/math?math=V"> is a 2D weight matrix, and <img src="https://render.githubusercontent.com/render/math?math=y"> is a one-hot encoding of the class label (where the conditional bias is broadcasted spacially and added channel-wise).


## Datasets
| Binary MNIST | Binary Shapes | Colorized MNIST | Colorized Shapes |
|:------------:|:-------------:|:---------------:|:----------------:|
|     ![](https://i.imgur.com/B1jo04C.png)         |     ![](https://i.imgur.com/fl3Qn81.png)          |     ![](https://i.imgur.com/VB7Dd8F.png)            |            ![](https://i.imgur.com/0phiKcu.png)      |


## Samples

| Model                    | MNIST Samples | Shapes Samples |
|--------------------------|:-------------:|:--------------:|
| PixelCNN                 |      ![](https://i.imgur.com/JJXwUOW.png)         |      ![](https://i.imgur.com/zhlpdi9.png)          |
| PixelRCNN                |      ![](https://i.imgur.com/UfvPcCM.png)         |      ![](https://i.imgur.com/mfr2jkf.png)          |
| Autoregressive PixelRCNN |      ![](https://i.imgur.com/6HH2TkU.png)         |      ![](https://i.imgur.com/emijyeN.png)          |
| Conditional PixelCNN     |      ![](https://i.imgur.com/pNuMZd2.png)         |      ![](https://i.imgur.com/TCZps2Y.png)          |
