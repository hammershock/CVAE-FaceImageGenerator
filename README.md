# Conditional Convolutional Variational Autoencoder (CVAE)

CVAE Image generator, implementation relies solely on NumPy

This project implements a NumPy-based Conditional Convolutional Variational Autoencoder (CVAE) trained on the CelebA Faces dataset. The implementation is designed to operate with minimal dependencies, specifically using only NumPy and matplotlib for the core functionality. PyTorch is optional and is used only for model training.

## Features

- **highly lightweight**, with a model size of 7.5 MB, the size reduced to ONLY 1.7MB (decoder only and fp16 quantized)
- the NumPy implementation is closely mimics the behavior of native PyTorch.

![image](https://github.com/hammershock/CVAEGenerator/assets/109429530/da2e55ca-a146-4728-bc90-8db30e1844b4)


## Requirements

- **NumPy**
- **matplotlib**
- **PyTorch** (optional, for training the model)
- **torchvision** (optional, for handling the CelebA dataset during training)

## Installation

1. Clone the repository and initialize submodules:
   ```bash
   git clone https://github.com/hammershock/CVAEGenerator.git
   cd CVAEGenerator
   git submodule init
   git submodule update
   ```
   
2. install the required packages using pip:

```bash
pip install numpy matplotlib
pip install torch torchvision  # For training only
```

## Usage

- Run `main.py` to execute a pre-trained model and visualize outputs.
- Use `train.py` to train a new model or continue training an existing model.
- Just Enjoy it!

## Data

The CelebA dataset is used in this project. Ensure you have downloaded and properly placed the dataset in the expected directory. You can find the dataset at [CelebA Dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).


- The model integrates image attributes into its architecture, making it capable of generating conditioned facial images.
- The system provides support for both loading pre-trained weights and training from scratch using PyTorch.

  
Feel free to reach out for any queries or suggestions regarding this implementation.
