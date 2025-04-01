# Test IV. Diffusion Models (Gravitational Lensing Image Generation)

The objective of this task is to develop generative models capable of producing high-quality gravitational lensing images that closely resemble real astronomical observations. We evaluated two advanced generative modeling techniques for this task and assessed their performance using standard evaluation metrics.

## Dataset Description

The dataset consists of simulated gravitational lensing images generated using astrophysical models. These images are grayscale and normalized to represent intensity values.

### Dataset Characteristics
- **Format**: NumPy (.npy) files containing grayscale images
- **Resolution**: 64×64 pixels
- **Normalization**: Images were normalized to the appropriate ranges ([-1, 1] for diffusion models, [0, 1] for GANs)

## Model Architecture Selection

For this gravitational lensing image generation task, we implemented and evaluated two advanced generative frameworks:

| Model | Characteristics |
|-------|-----------------|
| **DDIM (Denoising Diffusion Implicit Models)** | 1. A deterministic variant of Denoising Diffusion Probabilistic Models (DDPM) |
|        | 2. Utilizes a U-Net backbone with time embedding and self-attention layers |
|        | 3. Memory-optimized implementation with gradient checkpointing to handle GPU constraints |
|        | 4. Faster generation compared to standard DDPM through implicit sampling |
| **GAN with Self-Attention** | 1. Generator: Transposed convolutional network with self-attention mechanisms |
|        | 2. Discriminator: PatchGAN-style network with spectral normalization |
|        | 3. Memory-efficient self-attention blocks for capturing global relationships |
|        | 4. Training ratio of 3:1 for discriminator:generator updates (based on empirical optimization) |

We also attempted to implement a standard DDPM but encountered persistent GPU memory limitations despite various optimization techniques.

## Implementation Details

### DDIM Implementation
- **Architecture**: U-Net with self-attention at multiple levels
- **Noise Schedule**: Improved cosine schedule (vs. linear)
- **Sampling Method**: DDIM sampling with 100 steps (vs. 1000 for standard DDPM)
- **Memory Optimization**:
  - Gradient checkpointing
  - Mixed precision training (FP16)
  - Smaller batch size with gradient accumulation

### GAN Implementation
- **Generator**: 
  - Four upsampling blocks using transposed convolutions
  - ReLU activations, batch normalization
  - Sigmoid activation for output layer
  
- **Discriminator**:
  - Convolutional layers with stride 2 for downsampling
  - LeakyReLU with slope 0.2
  - Batch normalization between layers
  - Sigmoid output for binary classification

### Hyperparameters
- **DDIM**:
  - Batch Size: 16 (effective batch size of 64 with gradient accumulation)
  - Learning Rate: 5e-5 with AdamW optimizer
  - Training Epochs: 30
  
- **GAN**:
  - Batch Size: 64
  - Learning Rate: 2e-4 with Adam optimizer (betas: 0.5, 0.999)
  - Training Epochs: 100
  - Discriminator:Generator Update Ratio: 3:1

### Addressing Technical Challenges
- Implemented memory-efficient self-attention with adaptive downsampling for large feature maps
- Added label smoothing (0.9 for real, 0.1 for fake) to improve GAN training stability
- Used exponential moving average (EMA) of generator weights for improved sample quality
- Processed evaluation metrics in small batches on CPU to avoid CUDA out-of-memory errors

## Results and Analysis

### Quantitative Evaluation

| Model | FID Score | Inception Score |
|-------|-----------|-----------------|
| **DDIM Sampling** | **197.7868** | 1.0893 ± 0.1342 |
| **GAN with Self-Attention** | 330.2589 | **1.1339 ± 0.0108** |

![DDIM Samples](https://github.com/XAheli/DeepLense_ML4SCI-GSoC25/blob/main/Specific%20Test%20IV%20Diffusion%20Model/DDIM%20Diffusion/Images/DDIM%20Samples.png)

![GAN Quality Progression](https://github.com/XAheli/DeepLense_ML4SCI-GSoC25/blob/main/Specific%20Test%20IV%20Diffusion%20Model/GAN%20Self%20Attention/Images/quality_progression.gif)
<p align="left"><em>GAN Quality Progression</em></p>

FID scores measure the similarity between the distribution of generated images and real images, with lower values indicating better quality. Inception scores measure both quality and diversity, with higher values being better.

### Key Observations

1. **DDIM vs. GAN Performance**:
   - DDIM achieved significantly better FID scores (197.79 vs. 330.26), indicating better alignment with the real image distribution.
   - GAN with Self-Attention achieved slightly higher Inception Scores (1.13 vs. 1.09), suggesting potentially better diversity.

2. **Training Dynamics**:
   - GAN training showed characteristic adversarial dynamics: Generator loss remained steady (0.6-0.7) while Discriminator loss initially spiked (around 6.0) before stabilizing (2.6-2.7).
   - DDIM training showed more stable convergence with consistent loss reduction.

![GAN Training Dynamics](https://github.com/XAheli/DeepLense_ML4SCI-GSoC25/blob/main/Specific%20Test%20IV%20Diffusion%20Model/GAN%20Self%20Attention/Images/loss_curve.png)
<p align="center"><em>GAN Training Dynamics</em></p>

3. **Technical Limitations**:
   - Despite having 39.56 GiB of GPU memory, DDPM training consistently failed with out-of-memory errors.
   - DDIM sampling provided a practical alternative with significantly reduced memory requirements.

## Challenges Encountered

### Memory Constraints

![Error Details](https://github.com/XAheli/DeepLense_ML4SCI-GSoC25/blob/main/Specific%20Test%20IV%20Diffusion%20Model/DDIM%20Diffusion/Images/gpu%20limit%20error.png)
<p align="center"><em>Error Details</em></p>

DDPM implementation faced persistent CUDA out-of-memory errors. This occurred despite implementing:
   - Gradient checkpointing
   - Mixed precision training
   - Memory-efficient self-attention
   - Smaller batch sizes with gradient accumulation
   - Explicit GPU memory management (`torch.cuda.empty_cache()`)

### GAN Training Instability
- Initial GAN implementations suffered from mode collapse and training instability addressed through:
  - Label smoothing (0.9 for real, 0.1 for fake)
  - Adjusted discriminator:generator update ratio (3:1)
  - Spectral normalization in discriminator
  - Adam optimizer with betas (0.5, 0.999)

## Future Improvements

1. **Architecture Enhancements**:
 - Explore hierarchical diffusion models for better capture of multi-scale features
 - Implement transformer-based generators for potentially better global coherence
 - Investigate style-based GANs (StyleGAN) for improved attribute control

2. **Training Optimizations**:
 - Implement progressive growing for higher resolution outputs
 - Explore alternative diffusion formulations that require less memory
 - Implement curriculum learning for more stable GAN training

3. **Evaluation Methods**:
 - Develop physics-informed metrics specific to gravitational lensing
 - Incorporate perceptual metrics beyond FID and Inception Score
 - Evaluate model-based metrics like detectability by lensing-finding algorithms

---

*This work is part of the GSoC 2025 DeepLense evaluation task.*
