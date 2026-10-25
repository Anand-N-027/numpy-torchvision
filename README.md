Cat vs Dog Image Classification using PyTorch and TorchVision
Overview
This project implements a Convolutional Neural Network (CNN) from scratch using PyTorch for binary image classification of cats and dogs. The workflow includes image validation, preprocessing, dataset splitting, model development, training, and evaluation.

The project demonstrates fundamental computer vision and deep learning concepts, including:

Image preprocessing using TorchVision
Dataset cleaning and validation
Convolutional Neural Networks (CNNs)
Train, validation, and test set generation
Model training using PyTorch
Binary image classification
Apple Silicon GPU (MPS) support
Directory Structure
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
Description
dataset/PetImages

Contains image folders for the two target classes:

Cat
Dog
The dataset follows the structure required by torchvision.datasets.ImageFolder.

notebook.ipynb

Contains the complete pipeline including:

Dataset cleaning
Data transformation
Data loading
CNN architecture
Model training
Evaluation
Requirements
The project was developed using:

Python 3.11
PyTorch 2.2.2
TorchVision 0.17.2
NumPy 1.26.4
Pandas 2.3.3
Matplotlib 3.10.8
Pillow 11.3.0
Install dependencies:

pip install numpy pandas matplotlib torch torchvision pillow
About the Dataset
This project uses the Cats vs Dogs image dataset organized into separate directories for each class.

Classes
Cat
Dog
Dataset Statistics
Training Set  : 14,998 images
Validation Set: 4,999 images
Test Set      : 5,001 images
Total images used:

24,998
Data Preprocessing Pipeline
This section describes the preprocessing workflow used before model training.

1. Remove Corrupted Images
Images are validated using the Pillow library.

Image.open(path).verify()
Files that cannot be opened are automatically removed.

try:
    Image.open(path).verify()
except:
    os.remove(path)
Purpose
Removing corrupted files prevents DataLoader failures during training.

2. Image Transformation
All images are resized and converted into tensors.

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])
Output
Original Image
      ↓
128 × 128 Resize
      ↓
Tensor Conversion
3. Dataset Loading
The dataset is loaded using TorchVision's ImageFolder class.

dataset = datasets.ImageFolder(
    DATA_DIR,
    transform=transform
)
Detected classes:

['Cat', 'Dog']
4. Dataset Splitting
The dataset is split into:

60% Training
20% Validation
20% Testing
Implementation:

train_ds, val_ds, test_ds = random_split(
    dataset,
    [train_size, val_size, test_size]
)
Data Loading
PyTorch DataLoaders are used to batch and shuffle data.

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
Configuration
Batch Size: 32
CNN Architecture
The model consists of three convolutional blocks followed by a classifier.

Feature Extraction Layers
Input Image
(3 × 128 × 128)

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
Classification Layers
Flatten
Linear (16384 → 128)
ReLU
Linear (128 → 2)
Output Classes
Class 0 → Cat
Class 1 → Dog
Device Configuration
Training can be performed on:

CPU
device = torch.device("cpu")
Apple Silicon GPU (MPS)
device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
MPS support enables hardware acceleration on:

Apple M1
Apple M2
Apple M3
Apple M4
processors.

Model Training
Loss Function
criterion = nn.CrossEntropyLoss()
Optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)
Configuration
Epochs        : 10
Batch Size    : 32
Learning Rate : 0.001
Optimizer     : Adam
Loss Function : CrossEntropyLoss
Training Results
Observed losses during training:

Epoch 1
Train Loss: 0.6382
Val Loss  : 0.5703
Test Loss : 0.5794

Epoch 2
Train Loss: 0.5419
Val Loss  : 0.5275
Test Loss : 0.5229

Epoch 3
Train Loss: 0.4829
Val Loss  : 0.4846
Test Loss : 0.4841

Epoch 4
Train Loss: 0.4281
Val Loss  : 0.4598
Test Loss : 0.4636

Epoch 5
Train Loss: 0.3771
Val Loss  : 0.4657
Test Loss : 0.4855

Epoch 6
Train Loss: 0.3246
Val Loss  : 0.4706
Test Loss : 0.4895

Epoch 7
Train Loss: 0.2541
Val Loss  : 0.6071
Test Loss : 0.6110

Epoch 8
Train Loss: 0.1806
Val Loss  : 0.6309
Test Loss : 0.6654

Epoch 9
Train Loss: 0.1135
Val Loss  : 0.6942
Test Loss : 0.7342

Epoch 10
Train Loss: 0.0634
Val Loss  : 0.9392
Test Loss : 0.9725
Interpretation of Results
The training loss decreases consistently throughout training:

0.6382 → 0.0634
This indicates the model is learning features that distinguish cat and dog images.

However, validation and test losses begin increasing after approximately Epoch 5–6.

This suggests:

Model overfitting
Reduced generalization performance
Need for regularization techniques
Potential improvements include:

Data augmentation
Dropout layers
Early stopping
Batch normalization
Transfer learning using pretrained models
Generated Outputs
Training Outputs
Model Predictions
Training Loss
Validation Loss
Test Loss
Dataset Outputs
Cleaned Image Dataset
Train Dataset
Validation Dataset
Test Dataset
Future Improvements
Model Improvements
Add Dropout layers
Add Batch Normalization
Implement Early Stopping
Save best-performing model checkpoints
Evaluation Improvements
Classification accuracy
Precision
Recall
F1-score
Confusion matrix
ROC-AUC analysis
Computer Vision Improvements
Random flips
Random rotations
Color jittering
Data augmentation pipelines
Deep Learning Improvements
ResNet-18
ResNet-50
EfficientNet
MobileNet
Learning Outcomes
This project provides practical experience with:

Image classification
Deep learning fundamentals
Convolutional Neural Networks
Computer vision pipelines
PyTorch model development
Dataset preprocessing
DataLoader workflows
GPU acceleration using Apple Silicon
Model evaluation and overfitting analysis
