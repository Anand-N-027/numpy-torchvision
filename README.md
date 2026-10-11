A PyTorch notebook for cat-and-dog image classification using a custom convolutional neural network, with image verification, preprocessing, train/validation/test splits, and per-epoch loss tracking.

- Cat and Dog Image Classification with PyTorch
A notebook-based image classification project that trains a custom CNN to distinguish between cats and dogs. Built with PyTorch and torchvision, it includes dataset preparation, image preprocessing, model training, and loss evaluation.

Overview
The notebook implements the following workflow:

Verify images in the Cat and Dog folders and delete files that fail verification.

Resize images to 128 × 128 and convert them to tensors.

Load the dataset using torchvision.datasets.ImageFolder.

Randomly divide the dataset into training, validation, and test subsets.

Train a three-block convolutional neural network for 10 epochs.

Print training, validation, and test losses after each epoch.

The recorded training run uses the CPU. The notebook also contains an optional Apple GPU device-selection expression using MPS.
