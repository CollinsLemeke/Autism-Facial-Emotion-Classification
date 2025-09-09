## Autism Facial Emotion Classification
### Project Overview

This project applies Computer Vision and Deep Learning techniques to classify facial emotions of autistic children.
The motivation is to explore how AI can support autism research by identifying emotional expressions, which may assist in improving social communication and awareness.

Using a VGG16-based Convolutional Neural Network (CNN) with transfer learning, the model achieves ~70% accuracy on the test set.

### Dataset

The dataset used is from:
Autistic Children Emotions – Dr. Fatma M. Talaat (available on Kaggle
).

Images are organized into Train/Test folders.

Classes represent different emotions displayed by autistic children.

Training data is augmented to improve generalization.

### Architecture & Approach

🔹 Preprocessing

Resizing all images to 256x256.

Data Augmentation:

Rotation, shift, zoom, brightness adjustments.

Horizontal flipping for better generalization.

Normalization (rescale 1./255).

🔹 Model

Base Model: VGG16
 (pre-trained on ImageNet).

Custom Layers Added:

Flatten

Dense (ReLU)

Dropout

Dense (Softmax) for classification.

Loss Function: Categorical Crossentropy.

Optimizer: Adam.

Metrics: Accuracy.

### Tools & Libraries

TensorFlow / Keras
 – Deep Learning Framework

NumPy
 – Numerical Computation

Matplotlib
 – Visualization

Kaggle
 – Dataset source and environment

### Results

Training accuracy reached ~70% on the test dataset.

Model successfully differentiates between multiple facial emotions of autistic children.

### Example Workflow

Data Loading → Images loaded from train/test directories.

Preprocessing → Resized, augmented, normalized.

Model Training → VGG16 + custom layers.

Evaluation → Achieved 70% accuracy.

Visualization → Training/Validation accuracy and loss curves.

### Future Improvements

Fine-tuning deeper layers of VGG16.

Try modern architectures (ResNet, EfficientNet).

Use Grad-CAM / Explainable AI for better interpretability.

Expand dataset for more balanced training.
