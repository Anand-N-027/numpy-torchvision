# Cat vs Dog Image Classification using PyTorch and TorchVision

## Overview

This project implements a Convolutional Neural Network (CNN) from scratch using PyTorch for binary image classification of cats and dogs. The workflow includes image validation, preprocessing, dataset splitting, model development, training, and evaluation.

The project demonstrates fundamental computer vision and deep learning concepts, including:

- Image preprocessing using TorchVision
- Dataset cleaning and validation
- Convolutional Neural Networks (CNNs)
- Train, validation, and test set generation
- Model training using PyTorch
- Binary image classification
- Apple Silicon GPU (MPS) support

---

## Directory Structure

```text
.
├── notebook.ipynb
├── README.md
└── dataset/
    └── PetImages/
        ├── Cat/
        │   ├── 0.jpg
        │   ├── 1.jpg
        │   └── ...
        └── Dog/
            ├── 0.jpg
            ├── 1.jpg
            └── ...
```

---

## Dataset Description

### dataset/PetImages

Contains image folders for the two target classes:

- Cat
- Dog

The dataset follows the directory structure required by `torchvision.datasets.ImageFolder`.

### notebook.ipynb

Contains the complete machine learning pipeline:

- Dataset cleaning
- Data transformation
- Data loading
- CNN architecture
- Model training
- Evaluation

---

## Requirements

The project was developed using:

- Python 3.11
- PyTorch 2.2.2
- TorchVision 0.17.2
- NumPy 1.26.4
- Pandas 2.3.3
- Matplotlib 3.10.8
- Pillow 11.3.0

Install dependencies:

```bash
pip install numpy pandas matplotlib torch torchvision pillow
```

---

## About the Dataset

This project uses the Cats vs Dogs image dataset organized into separate directories for each class.

### Classes

- Cat
- Dog

### Dataset Statistics

| Dataset Split | Images |
|--------------|---------|
| Training Set | 14,998 |
| Validation Set | 4,999 |
| Test Set | 5,001 |
| **Total** | **24,998** |

---

## Data Preprocessing Pipeline

### 1. Remove Corrupted Images

Images are validated using the Pillow library.

```python
Image.open(path).verify()
```

Files that cannot be opened are automatically removed.

```python
try:
    Image.open(path).verify()
except:
    os.remove(path)
```

#### Purpose

Removing corrupted files prevents DataLoader failures during training.

---

### 2. Image Transformation

All images are resized and converted into tensors.

```python
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])
```

#### Output

```text
Original Image
      ↓
128 × 128 Resize
      ↓
Tensor Conversion
```

---

### 3. Dataset Loading

The dataset is loaded using TorchVision's `ImageFolder` class.

```python
dataset = datasets.ImageFolder(
    DATA_DIR,
    transform=transform
)
```

Detected classes:

```python
['Cat', 'Dog']
```

---

### 4. Dataset Splitting

The dataset is split into:

- 60% Training
- 20% Validation
- 20% Testing

Implementation:

```python
train_ds, val_ds, test_ds = random_split(
    dataset,
    [train_size, val_size, test_size]
)
```

---

## Data Loading

PyTorch DataLoaders are used to batch and shuffle data.

```python
train_loader = DataLoader(
    train_ds,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=32,
    shuffle=False
)

test_loader = DataLoader(
    test_ds,
    batch_size=32,
    shuffle=False
)
```

### Configuration

- Batch Size: 32

---

## CNN Architecture

The model consists of three convolutional blocks followed by a classifier.

### Feature Extraction Layers

```text
Input Image (3 × 128 × 128)
        │
        ├── Conv2D (3 → 16)
        ├── ReLU
        ├── MaxPool
        │
        ├── Conv2D (16 → 32)
        ├── ReLU
        ├── MaxPool
        │
        ├── Conv2D (32 → 64)
        ├── ReLU
        └── MaxPool
```

### Classification Layers

```text
Flatten
   ↓
Linear (16384 → 128)
   ↓
ReLU
   ↓
Linear (128 → 2)
```

### Output Classes

- Class 0 → Cat
- Class 1 → Dog

---

## Device Configuration

Training can be performed on:

### CPU

```python
device = torch.device("cpu")
```

### Apple Silicon GPU (MPS)

```python
device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)
```

MPS support enables hardware acceleration on:

- Apple M1
- Apple M2
- Apple M3
- Apple M4

---

## Model Training

### Loss Function

```python
criterion = nn.CrossEntropyLoss()
```

### Optimizer

```python
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)
```

### Configuration

- Epochs: 10
- Batch Size: 32
- Learning Rate: 0.001
- Optimizer: Adam
- Loss Function: CrossEntropyLoss

---

## Training Results

| Epoch | Train Loss | Validation Loss | Test Loss |
|---------|---------|---------|---------|
| 1 | 0.6382 | 0.5703 | 0.5794 |
| 2 | 0.5419 | 0.5275 | 0.5229 |
| 3 | 0.4829 | 0.4846 | 0.4841 |
| 4 | 0.4281 | 0.4598 | 0.4636 |
| 5 | 0.3771 | 0.4657 | 0.4855 |
| 6 | 0.3246 | 0.4706 | 0.4895 |
| 7 | 0.2541 | 0.6071 | 0.6110 |
| 8 | 0.1806 | 0.6309 | 0.6654 |
| 9 | 0.1135 | 0.6942 | 0.7342 |
| 10 | 0.0634 | 0.9392 | 0.9725 |

---

## Interpretation of Results

The training loss decreases consistently throughout training:

```text
0.6382 → 0.0634
```

This indicates that the model is successfully learning features that distinguish cat and dog images.

However, validation and test losses begin increasing after approximately Epochs 5–6.

This suggests:

- Model overfitting
- Reduced generalization performance
- Need for regularization techniques

Potential improvements include:

- Data augmentation
- Dropout layers
- Early stopping
- Batch normalization
- Transfer learning using pretrained models

---

## Generated Outputs

### Training Outputs

- Model Predictions
- Training Loss
- Validation Loss
- Test Loss

### Dataset Outputs

- Cleaned Image Dataset
- Train Dataset
- Validation Dataset
- Test Dataset

---

## Future Improvements

### Model Improvements

- Add Dropout layers
- Add Batch Normalization
- Implement Early Stopping
- Save best-performing model checkpoints

### Evaluation Improvements

- Classification Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC-AUC Analysis

### Computer Vision Improvements

- Random Flips
- Random Rotations
- Color Jittering
- Data Augmentation Pipelines

### Deep Learning Improvements

- ResNet-18
- ResNet-50
- EfficientNet
- MobileNet

---

## Learning Outcomes

This project provides practical experience with:

- Image Classification
- Deep Learning Fundamentals
- Convolutional Neural Networks (CNNs)
- Computer Vision Pipelines
- PyTorch Model Development
- Dataset Preprocessing
- DataLoader Workflows
- GPU Acceleration using Apple Silicon
- Model Evaluation and Overfitting Analysis

---

## License

This project is intended for educational and learning purposes.
