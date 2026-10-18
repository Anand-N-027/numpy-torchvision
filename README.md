Features
Automatic detection and removal of corrupted images
Image preprocessing using TorchVision transforms
Custom CNN architecture built from scratch
Train/Validation/Test dataset splitting
Model training using PyTorch
Support for Apple Silicon (MPS) and CPU devices
Performance monitoring through training, validation, and test losses
Technologies Used
Python
PyTorch
TorchVision
NumPy
Pandas
Pillow (PIL)
Matplotlib
Jupyter Notebook
Dataset Structure
The project expects the dataset to be organized as follows:

PetImages/
│
├── Cat/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
│
└── Dog/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
Update the dataset path in the notebook before training:

DATA_DIR = "/path/to/PetImages"
Data Preprocessing
Before training, the notebook:

Removes corrupted image files
Resizes images to 128 × 128
Converts images into tensors
Splits the dataset into:
60% Training
20% Validation
20% Testing
CNN Architecture
Feature Extraction Layers
Conv2D (3 → 16)
ReLU
MaxPool

Conv2D (16 → 32)
ReLU
MaxPool

Conv2D (32 → 64)
ReLU
MaxPool
Classification Layers
Flatten
Linear (16384 → 128)
ReLU
Linear (128 → 2)
Output Classes:

Cat
Dog
Training Configuration
Batch Size = 32
Epochs = 10
Optimizer = Adam
Learning Rate = 0.001
Loss Function = CrossEntropyLoss
Training Results
Epoch [1/10]  | Train Loss: 0.6382
Epoch [2/10]  | Train Loss: 0.5419
Epoch [3/10]  | Train Loss: 0.4829
Epoch [4/10]  | Train Loss: 0.4281
Epoch [5/10]  | Train Loss: 0.3771
Epoch [6/10]  | Train Loss: 0.3246
Epoch [7/10]  | Train Loss: 0.2541
Epoch [8/10]  | Train Loss: 0.1806
Epoch [9/10]  | Train Loss: 0.1135
Epoch [10/10] | Train Loss: 0.0634
The model shows a significant reduction in training loss, indicating successful learning of cat and dog image features.

Installation
Clone the repository:

git clone https://github.com/Anand-N-027/numpy-torchvision.git
cd numpy-torchvision
Install dependencies:

pip install numpy pandas matplotlib torch torchvision pillow
Running the Project
Launch Jupyter Notebook:

jupyter notebook
Open:

Untitled8.ipynb
Run all cells sequentially to preprocess the dataset and train the model.

Apple Silicon Support
For Mac users with M-series chips, the notebook includes support for Metal Performance Shaders (MPS):

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)
This enables GPU acceleration on Apple Silicon devices.

Learning Outcomes
This project helped me gain hands-on experience with:

Deep Learning Fundamentals
Convolutional Neural Networks (CNNs)
Image Classification
Dataset Preprocessing
PyTorch Model Development
Computer Vision Workflows
GPU Acceleration with Apple Silicon
Future Improvements
Add accuracy and F1-score metrics
Implement data augmentation
Save and load trained models
Generate confusion matrices
Use transfer learning with ResNet/EfficientNet
Create an image prediction interface
Deploy the model as a web application
Author
Anand Narendran

GitHub: https://github.com/Anand-N-027

