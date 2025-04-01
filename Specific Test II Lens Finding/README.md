# Test II. Lens Finding

Gravitational lensing occurs when a massive object (such as a galaxy) bends the path of light from a distant source as it travels toward an observer. Strong gravitational lensing can produce multiple images, arcs, or even complete rings (Einstein rings) of the background source. The detection of gravitational lenses is crucial for studying dark matter distribution and cosmic structure.

The objective of this task is to develop a model capable of accurately classifying images as either containing gravitational lensing effects (lens) or not containing them (non-lens). This binary classification challenge is particularly difficult due to the extreme class imbalance present in real astronomical surveys, where lensing events are rare.

## Dataset Description

The dataset consists of simulated astronomical images with and without gravitational lensing effects. The dataset exhibits significant class imbalance that reflects real-world conditions.

### Dataset Structure
- Training set: 30,405 images (1,730 lens, 28,675 non-lens)
- Test set: 19,650 images (195 lens, 19,455 non-lens)
- Format: NumPy (.npy) files with shape (3, 64, 64)
- Class imbalance: ~1:16 in training, ~1:100 in testing

```
dataset/
│
├── train_lenses/
│   └── [1,730 .npy files]
│
├── train_nonlenses/
│   └── [28,675 .npy files]
│
├── test_lenses/
│   └── [195 .npy files]
│
└── test_nonlenses/
    └── [19,455 .npy files]
```

## Model Architecture Selection

For this highly imbalanced binary classification task, we evaluated multiple architectures:

1. **ResNet18**: A residual network architecture that utilizes skip connections to address the vanishing gradient problem. Its moderate depth and parameter count make it well-suited for the 64×64 input images.

2. **EfficientNet (B1/B2)**: Models designed for efficiently scaling networks in a structured way. They offer a good balance between model size and performance.

3. **MobileViT**: A hybrid architecture combining CNN efficiency with transformer attention mechanisms, offering potentially better feature extraction for complex patterns.

4. **Ensemble Model**: A combination of models (EfficientNet-B1 and B2) to leverage complementary strengths and improve overall performance.

ResNet18 demonstrated superior performance, likely due to its residual architecture being particularly effective for detecting the subtle distortion patterns in gravitational lensing images at 64×64 resolution.

## Implementation Details

### Data Preprocessing and Augmentation
- Strong data augmentation on the lens class (minority class) using albumentations
- Channel-wise normalization based on dataset statistics
- 90:10 train-validation split from the original training data

### Hyperparameters
- **Batch Size**: 32
- **Learning Rate**: 1e-4
- **Weight Decay**: 1e-4
- **Training Epochs**: 15
- **Optimizer**: AdamW
- **Loss Function**: BCE with Logits Loss with class weighting
- **Early Stopping**: Patience of 5 epochs

### Addressing Class Imbalance
- Aggressive data augmentation of the minority class
- Class weighting in the loss function
- Threshold optimization for F1-score (rather than using default 0.5)

## Results and Analysis

### Comprehensive Model Performance Comparison

| Model | AUC Score | F1 Score | Precision | Recall |
|-------|-----------|----------|-----------|--------|
| **ResNet18** | 0.98 | 0.21 | 0.12 | 0.95 |
| **Ensemble** | 0.95 | 0.11 | 0.06 | 0.94 |
| **MobileViT** | 0.94 | 0.15 | 0.08 | 0.84 |
| **EfficientNet-B1** | 0.94 | 0.12 | 0.07 | 0.87 |
| **EfficientNet-B2** | 0.93 | 0.08 | 0.05 | 0.94 |

The models achieved excellent AUC scores (0.93-0.98), demonstrating strong discriminative ability. However, the relatively low F1 scores (0.08-0.21) reveal the challenge of balancing precision and recall in this highly imbalanced dataset.

ResNet18 outperformed all other architectures across both metrics. With an optimal threshold, it achieved high recall (0.95) while maintaining better precision than other models, resulting in the best F1 score (0.21).

### Key Observations

1. **High AUC vs. Low F1**: All models show excellent ranking ability (AUC > 0.9) but struggle with the precision-recall trade-off, a common challenge in highly imbalanced datasets.

2. **Threshold Optimization**: Using customized classification thresholds (rather than the default 0.5) significantly improved model performance, particularly for recall of the minority class.

3. **Architecture Performance**: ResNet18's superior performance suggests that its residual connections are particularly effective at capturing the subtle features of gravitational lensing at the given resolution.

4. **Ensemble Benefits**: While the ensemble model achieved competitive AUC, it did not outperform the single ResNet18 model, suggesting that model diversity did not provide additional benefits for this particular task.

## Results Visualization

### Model Comparison
![Model Comparison](https://github.com/XAheli/DeepLense_ML4SCI-GSoC25/blob/main/Specific%20Test%20II%20Lens%20Finding/Images/model_comparison.png)
![Best Model Training History](https://github.com/XAheli/DeepLense_ML4SCI-GSoC25/blob/main/Specific%20Test%20II%20Lens%20Finding/Images/best%20model_training%20history.png)
![Best Model Confusion Matrix](https://github.com/XAheli/DeepLense_ML4SCI-GSoC25/blob/main/Specific%20Test%20II%20Lens%20Finding/Images/best%20model_confusion%20Matrix.png)
![Best Model ROC](https://github.com/XAheli/DeepLense_ML4SCI-GSoC25/blob/main/Specific%20Test%20II%20Lens%20Finding/Images/best%20model_ROC.png)


This project demonstrates effective approaches for gravitational lens detection in highly imbalanced datasets. ResNet18 emerged as the superior architecture for this task, achieving an AUC score of 0.98 and F1 score of 0.21. The significant gap between AUC and F1 highlights the inherent challenge in balancing precision and recall for rare object detection in astronomical data.

Key contributions include:
1. A comprehensive evaluation of modern deep learning architectures for lens detection
2. Effective strategies for handling extreme class imbalance (up to 1:100)
3. Threshold optimization techniques that significantly improve minority class detection

Future work could explore more advanced augmentation techniques, self-supervised pretraining approaches, and attention-based mechanisms specifically designed for small, subtle features that characterize gravitational lensing phenomena.

---

*This work is done as part of the GSoC 2025 DeepLense evaluation task.*
