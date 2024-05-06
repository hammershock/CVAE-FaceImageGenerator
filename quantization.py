"""
Format convert & Model Quantization

example 1. convert torch format .pth to fake torch format .pkl
example 2. quantize the model to fp16 to reduce the model size,
the size of the decoder-only model at fp16 ONLY takes about 1.7MB!
"""

import numpy as np
import pickle

import NumPyTorch as torch  # fake torch

if __name__ == '__main__':
    # state_dict = torch.load_pytorch('./models/cvae_celeba.pth')  # load from torch model, you need to install pytorch.
    # torch.save('./models/cvae_celeba.pkl', state_dict)

    state_dict = torch.load('./models/cvae_celeba.pkl')
    print(state_dict.keys())
    filtered_state_dict = {key: value for key, value in state_dict.items() if 'enc' not in key}
    print(filtered_state_dict.keys())
    torch.save(filtered_state_dict, './models/cvae_celeba_decoder_only.pkl')

    filtered_state_dict2 = {key: value.astype(np.float16) for key, value in filtered_state_dict.items()}
    torch.save(filtered_state_dict2, './models/cvae_celeba_decoder_only_fp16.pkl')

