# CVAE-FaceImageGenerator

A face image generator based on a Conditional Convolutional Variational Autoencoder (CVAE) implemented solely with `numpy`.

[中文文档](./README.md)

## Features

- **Highly lightweight**: DecoderOnly and fp16 quantized model size is only 1.7MB
- `numpy` implementation closely mimics PyTorch interface
- Supports image display using colored characters (ASCII Art)

![myplot.png](assets/myplot.png)

## Dependencies

- **numpy**
- **matplotlib**
- **PyTorch** (optional, for training the model)
- **torchvision** (optional, for handling the CelebA dataset during training)

## Installation

1. Clone the repository and initialize submodules:
   ```bash
   git clone --recurse-submodules https://github.com/hammershock/CVAEGenerator.git
   cd CVAEGenerator
   ```

2. Install the required packages using pip:
   ```bash
   pip install numpy matplotlib
   pip install torch torchvision  # For training only
   ```

## Usage

- Run `main.py` for inference to generate face images using the pre-trained model.
- Train the model with `train.py`.

## Data

This project uses the [CelebA dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

- The model integrates image attributes into its architecture, enabling conditional face image generation.
- The system supports loading pre-trained weights and training from scratch using PyTorch.