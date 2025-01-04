# CNN for COVIDVSBACTERIA 
## COVID-19 and Bacteria Detection System Using CNN

## Project Overview
This project implements a deep learning system designed to detect and classify medical images into three categories: COVID-19, bacterial infection, and healthy cases. The system utilizes a Convolutional Neural Network (CNN) architecture to analyze medical imaging data, specifically chest X-rays, to assist in rapid diagnosis.

## Data Characteristics
- **Input Data**: Medical images (X-rays) in PNG, JPG, or JPEG format
- **Image Dimensions**: Standardized to 224x224 pixels with 3 color channels
- **Classes**: Three distinct categories
  - COVID-19 positive cases
  - Bacterial infection cases
  - Healthy cases
- **Dataset Split**: 80% training data, 20% testing data

## System Architecture

### Data Preprocessing
1. Image Loading and Standardization
   - Images are loaded from the dataset directory
   - Resized to uniform dimensions (224x224)
   - Pixel values normalized to range [0,1]

2. Label Encoding
   - Categorical labels converted to one-hot encoded format
   - Three-dimensional output vector corresponding to three classes

### CNN Model Architecture
- **Input Layer**: Accepts 224x224x3 dimensional images
- **Convolutional Layers**:
  - First Conv2D: 32 filters with 3x3 kernel
  - Second Conv2D: 64 filters with 3x3 kernel
- **Pooling Layers**: MaxPooling2D with 2x2 pool size
- **Regularization**: Dropout layer (0.3 rate) to prevent overfitting
- **Dense Layers**:
  - First Dense: 128 neurons with ReLU activation
  - Second Dense: 64 neurons with ReLU activation
  - Output Dense: 3 neurons with Softmax activation

## Training Configuration
- **Optimizer**: Adam
- **Loss Function**: Categorical Cross-entropy
- **Batch Size**: 16
- **Epochs**: 5
- **Validation**: Real-time validation using test set

## Performance Metrics
The system achieves impressive performance metrics:
- **Overall Accuracy**: 96.24%
- **Precision**: 96.58%
- **Recall**: 96.43%
- **F1-Score**: 96.44%

## Visualization Components
1. **Training Visualization**:
   - Display of sample images with true labels
   - Real-time training progress monitoring

2. **Evaluation Visualization**:
   - Confusion matrix for performance analysis
   - Random sample predictions with true vs predicted labels

## System Features
1. **Automated Processing**:
   - Automatic directory traversal
   - Dynamic file format handling
   - Batch processing capability

2. **Robust Architecture**:
   - LeakyReLU activation for better gradient flow
   - Dropout layers for regularization
   - Batch normalization support

3. **Performance Analysis**:
   - Comprehensive metrics calculation
   - Visual performance representation
   - Detailed confusion matrix

## Technical Requirements
- Python 3.x
- Key Libraries:
  - TensorFlow/Keras
  - NumPy
  - Pandas
  - Scikit-learn
  - OpenCV
  - Matplotlib

## Practical Applications
1. **Medical Diagnosis Assistance**:
   - Rapid screening of chest X-rays
   - Early detection support
   - Triage assistance in high-volume settings

2. **Research Support**:
   - Pattern identification in medical imaging
   - Comparative analysis of different pathologies
   - Data-driven insights for clinical research

## Limitations and Considerations
1. **Data Dependencies**:
   - Requires high-quality medical imaging
   - Balanced dataset for optimal performance
   - Proper image formatting and standardization

2. **Processing Requirements**:
   - Computational resources for training
   - Memory requirements for batch processing
   - Storage considerations for dataset management

## Future Enhancement Possibilities
1. **Model Improvements**:
   - Additional convolutional layers
   - Advanced regularization techniques
   - Dynamic learning rate adjustment

2. **Feature Expansion**:
   - Multi-modal image support
   - Real-time processing capabilities
   - Integration with medical systems

3. **Performance Optimization**:
   - Model compression techniques
   - Inference speed optimization
   - Resource usage optimization
