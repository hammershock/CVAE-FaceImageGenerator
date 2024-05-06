import numpy as np

import NumPyTorch as torch

if __name__ == '__main__':
    state_dict = torch.load('./models/cvae_celeba.pkl')
    print(state_dict.keys())
    filtered_state_dict = {key: value for key, value in state_dict.items() if 'enc' not in key}
    print(filtered_state_dict.keys())
    torch.save(filtered_state_dict, './models/cvae_celeba_decoder_only.pkl')

    filtered_state_dict2 = {key: value.astype(np.float16) for key, value in filtered_state_dict.items()}
    torch.save(filtered_state_dict2, './models/cvae_celeba_decoder_only_fp16.pkl')
