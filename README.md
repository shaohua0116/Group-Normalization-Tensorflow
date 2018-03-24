# Group Normalization

## Descriptions
This project includes a Tensorflow implementation of Group Normalizations proposed in the paper 
[Group Normalization](https://arxiv.org/abs/1803.08494) by Wu et al. 
Batch Normalization (BN) has been widely employed in the trainings of deep neural networks to alleviate the internal covariate shift [1].
Specifically, BN aims to transform the inputs of each layer in such a way that they have a mean output activation of zero and standard deviation of one. While BN demonstrates it effectiveness in a variety of fields including computer vision, natural language processing, speech processing, robotics, etc., BN's performance substantially decrease when the training batch size become smaller, which limits the gain of utilizing BN in a task requiring small batches constrained by memory consumption. Motivated by this phenomenon, the Group Normalization (GN) technique is proposed. Instead of normalizing along the batch dimension, GN divides the channels into groups and computes within each group the mean and variance. Therefore, GN’s computation is independent of batch sizes, and so does its accuracy. The experiment section of the paper demonstrates the effectiveness of GN in a wide range of visual tasks, which include image classification (ImageNet), object detection and segmentation (COCO), and video classification (Kinect). This repository is simply a toy repository for those who want to quickly test GN and compare it against BN.

## Prerequisites

- Python 2.7 or Python 3.3+
- [Tensorflow 1.3.0](https://github.com/tensorflow/tensorflow/tree/r1.0)
- [SciPy](http://www.scipy.org/install.html)
- [NumPy](http://www.numpy.org/)

## Usage

### Datasets
Download datasets with:
```bash
$ python download.py --dataset MNIST Fashion SVHN CIFAR10
```

### Train models with downloaded datasets:
Specify the type of normalization you want to use by `--norm_type batch` or `--norm_type group` 
and specify the batch size with `--batch_size BATCH_SIZE`.
```bash
$ python trainer.py --dataset MNIST --learning_rate 1e-3
$ python trainer.py --dataset Fashion --prefix test
$ python trainer.py --dataset SVHN --batch_size 128
$ python trainer.py --dataset CIFAR10 
```

### Train and test your own datasets:

* Create a directory
```bash
$ mkdir datasets/YOUR_DATASET
```

* Store your data as an h5py file datasets/YOUR_DATASET/data.hy and each data point contains
    * 'image': has shape [h, w, c], where c is the number of channels (grayscale images: 1, color images: 3)
    * 'label': represented as an one-hot vector
* Maintain a list datasets/YOUR_DATASET/id.txt listing ids of all data points
* Modify trainer.py including args, data_info, etc.
* Finally, train and test models:
```bash
$ python trainer.py --dataset YOUR_DATASET
$ python evaler.py --dataset YOUR_DATASET
```

## Results

## Related works
* [Group Normalization](https://arxiv.org/abs/1803.08494)
* [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)

## Author

Shao-Hua Sun / [@shaohua0116](https://shaohua0116.github.io/) @ [Joseph Lim's research lab](https://github.com/gitlimlab) @ USC
