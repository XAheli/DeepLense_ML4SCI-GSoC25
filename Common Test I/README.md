# Common Test I. Multi-Class Classification

Gravitational lensing occurs when a massive object (such as a galaxy) bends the path of light from a distant source as it travels toward an observer. Strong gravitational lensing can produce multiple images, arcs or even complete rings (Einstein rings) of the background source. These lensing effects are influenced by the mass distribution within the lensing galaxy, including potential substructures.

The objective of this task is to develop a model capable of accurately classifying strong lensing images into three categories:
1. **No substructure**: Images showing lensing effects without distinctive substructure
2. **Subhalo substructure**: Images with lensing influenced by subhalos in the galaxy
3. **Vortex substructure**: Images with vortex-type substructure affecting the lensing pattern

Accurate classification of these images contributes to our understanding of dark matter distributions in galaxies and provides insights into cosmic structure formation.

## Dataset Description

The Dataset consists of three classes, strong lensing images with no substructure, subhalo substructure, and vortex substructure. The images have been normalized using min-max normalization.

### Dataset Structure
- Training set: 30,000 images 
- Validation set: 7,500 images
- Format: NumPy (.npy) files
- Preprocessing: Min-max normalization already applied

The dataset is organized in a structured format with separate directories for each class:
```
dataset/
│
├── train/
│   ├── no/
│   ├── sphere/
│   └── vort/
│
└── val/
    ├── no/
    ├── sphere/
    └── vort/
```

## Why DenseNet Architectures?

DenseNet architectures are particularly effective for gravitational lensing classification due to:

1. **Dense Connectivity**: The dense connectivity pattern, where each layer receives inputs from all preceding layers, helps preserve fine-grained features across the network. This is essential for detecting subtle substructures in lensing images.

2. **Feature Reuse**: DenseNets encourage feature reuse throughout the network, leading to more compact models with fewer parameters than comparable architectures, which is beneficial when working with limited training data.

3. **Gradient Flow**: The dense connections provide direct paths for gradient flow during backpropagation, alleviating the vanishing gradient problem and enabling effective training of deeper networks.

4. **Empirical Performance**: Previous studies have demonstrated that DenseNet architectures achieve state-of-the-art performance on various image classification tasks, including astronomical image analysis.


## Implementation Details

### Hyperparameters

- **Batch Size**: 64 for individual models
- **Learning Rate**: 1e-4
- **Dropout Rate**: 0.33
- **Training Epochs**: 10 for individual models, 10 for ensemble
- **Optimizer**: Adam
- **Loss Function**: Cross-Entropy Loss


## Results and Analysis

### Comprehensive Model Performance Comparison

| Metrics | DenseNet161 | DenseNet201 | Ensemble Model |
|---------|-------------|-------------|----------------|
| **Overall Accuracy** | 89-90% | 88-89% | 91-92% |
| **AUC (OVO)** | 0.91 | 0.90 | 0.94-0.95 |
| **AUC (OVR)** | 0.90 | 0.89 | 0.94-0.95 |
| **No Substructure Class** | Precision: 0.91Recall: 0.88F1: 0.90 | Precision: 0.90Recall: 0.87F1: 0.88 | Precision: 0.92Recall: 0.90F1: 0.91 |
| **Subhalo Substructure Class** | Precision: 0.87Recall: 0.89F1: 0.88 | Precision: 0.88Recall: 0.90F1: 0.89 | Precision: 0.90Recall: 0.92F1: 0.91 |
| **Vortex Substructure Class** | Precision: 0.89Recall: 0.85F1: 0.87 | Precision: 0.91Recall: 0.86F1: 0.88 | Precision: 0.91Recall: 0.89F1: 0.90 |
| **Per-Class AUC Range** | 0.90-0.92 | 0.88-0.91 | ~0.95 |
| **Confusion Matrix Analysis** | Minor misclassifications at class boundaries; strongest in "No Substructure" classification | Slightly different error pattern from DenseNet161; strongest in "Subhalo" classification | High diagonal dominance; significantly reduced misclassifications across all classes |
| **Best Performance Area** | No Substructure class (F1: 0.90) | Subhalo Substructure class (F1: 0.89) | Balanced across classes with slight edge in No Substructure (F1: 0.91) |
| **Combined Predictions AUC** | - | - | 0.95+ (when averaged with other models) |

*Note: The Combined Predictions row refers to averaging the probability outputs from all three models, which achieved AUC scores exceeding 0.95, demonstrating the complementary nature of the models' decision boundaries.*


## Results Visualization

### DenseNet161
![DenseNet161 Confusion Matrix](https://github.com/XAheli/DeepLense_ML4SCI-GSoC25/blob/main/Common%20Test%20I/Images/Densenet161/conf%20mat.png)
![DenseNet161 ROC Curve](https://github.com/XAheli/DeepLense_ML4SCI-GSoC25/blob/main/Common%20Test%20I/Images/Densenet161/download%20(1).png)
![DenseNet161 Accuracy and Loss](https://github.com/XAheli/DeepLense_ML4SCI-GSoC25/blob/main/Common%20Test%20I/Images/Densenet161/download.png)
![DenseNet161 Classification Report](https://github.com/XAheli/DeepLense_ML4SCI-GSoC25/blob/main/Common%20Test%20I/Images/Densenet161/Screenshot%202025-03-30%20025552.png)

### DenseNet201
![DenseNet201 Confusion Matrix](https://github.com/XAheli/DeepLense_ML4SCI-GSoC25/blob/main/Common%20Test%20I/Images/Densenet201/conf%20matrix.png)
![DenseNet201 ROC Curve](https://github.com/XAheli/DeepLense_ML4SCI-GSoC25/blob/main/Common%20Test%20I/Images/Densenet201/download.png)
![DenseNet201 Accuracy and Loss](https://github.com/XAheli/DeepLense_ML4SCI-GSoC25/blob/main/Common%20Test%20I/Images/Densenet201/download%20(2).png)
![DenseNet201 Classification Report](https://github.com/XAheli/DeepLense_ML4SCI-GSoC25/blob/main/Common%20Test%20I/Images/Densenet201/Screenshot%202025-03-30%20025858.png)

### Ensemble
![Ensemble Model Confusion Matrix](https://github.com/XAheli/DeepLense_ML4SCI-GSoC25/blob/main/Common%20Test%20I/Images/Ensemble/download.png)
![Ensemble Model ROC Curve](https://github.com/XAheli/DeepLense_ML4SCI-GSoC25/blob/main/Common%20Test%20I/Images/Ensemble/download%20(2).png)
![Ensemble Model Accuracy and Loss](https://github.com/XAheli/DeepLense_ML4SCI-GSoC25/blob/main/Common%20Test%20I/Images/Ensemble/download%20(1).png)
![Ensemble Model Classification Report](https://github.com/XAheli/DeepLense_ML4SCI-GSoC25/blob/main/Common%20Test%20I/Images/Ensemble/Screenshot%202025-03-30%20030114.png)

---

*This work is done as part of the GSoC 2025 DeepLense evaluation task.*

