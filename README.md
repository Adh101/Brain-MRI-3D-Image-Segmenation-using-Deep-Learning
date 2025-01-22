# Brain MRI 3D Image Segmentation Using U-Net and Attention U-Net

## Project Overview
This project focuses on the segmentation of 3D brain MRI images using Convolutional Neural Networks (CNNs). The primary models used are U-Net and Attention U-Net, which are designed to efficiently capture spatial hierarchies and segment medical images with high precision.

---

## Key Features
- **3D MRI Image Segmentation**: Preprocessing and segmentation of volumetric brain MRI data.
- **Model Architectures**: Implementation and comparison of U-Net and Attention U-Net for enhanced segmentation performance.

---

## Requirements

### Hardware
- **Kaggle P100 GPU** (or equivalent for training)

### Software and Libraries
- Python 3.8+
- Tensorflow
- Keras
- NumPy
- Matplotlib
- Scikit-learn
- SimpleITK
- nibabel
- tqdm

## Dataset
### Description
For this task, we are using MICCAI Brain Tumour
Segmentation Challenge 2020 (BraTS) dataset. Image
data descriptions All BraTS multimodal scans are available as
NIfTI files (.nii.gz),commonly used medical imaging format
to store brain imaging data obtained using MRI and describe
different MRI settings. We are using only training data and
splitting them into train and validation dataset for our experi-
ments.

Size of the dataset: 368 Brain MRI Scans Folders
• Type of Volumes: 4 different volumes of same re-
gion: Native(T1), Post- Contrast T1 weighted (T1ce), T2
Weighted(T2), T2 FLAIR
• Annotated Labels: Label 0(Unlabeled volume), Label
1(Necrotic and non-enhancing tumor core (NCR/NET)),
Label 2( Peritumoral edema (ED)), Label 3( Missing),
Label 4 ( GD-enhancing tumor (ET))

### Preprocessing
1. Scaling the 3D MRI Image(240 * 240 * 155) using Min Max Scaler.
2. Type Conversion: Masked image to uint8.
3. Combine: T2, T1CE, FLAIR Images (240 * 240 * 155 * 3).
4. Convert Masks to Categorical.
5. Crop the MRI Image to 128 * 128 * 128 * 3.

---

## Model Architectures

### U-Net
U-Net is a CNN architecture with an encoder-decoder structure and skip connections that help recover spatial information during upsampling.

### Attention U-Net
An extension of U-Net, integrating attention mechanisms to focus on relevant regions of the image for more precise segmentation.

---

## Training Pipeline
1. **Data Loading**: Created a loader function to read numpy array.
2. **Loss Function**: Total Loss ( Dice and Focal Loss)
3. **Optimizer**: Adam optimizer with a learning rate of 0.001.
4. **Scheduler**: Learning rate scheduler for dynamic adjustment.
5. **Training Specifications**:
   - Epochs: 10 and 30
   - Batch Size: 4
   - Sequence Length: 128

---

## Evaluation
1. **Metrics**:
   - Total Loss
   - Intersection over Union (IoU)

2. **Visualization**:
   - Plot segmentation masks alongside the input images.

---

## Results
### Performance Metrics
- Quantitative results based on Dice Coefficient and IoU.
- Comparative analysis of U-Net and Attention U-Net.

### Qualitative Results
- Visual comparisons of segmentation masks produced by both models.
---

## Future Scope
- Incorporating advanced architectures like Transformer-based models.
- Exploring semi-supervised or unsupervised learning techniques.
- Extending the approach to multimodal image segmentation.

---

## References
- [Original U-Net Paper](https://arxiv.org/abs/1505.04597)
- [Attention U-Net Paper](https://arxiv.org/abs/1804.03999)

