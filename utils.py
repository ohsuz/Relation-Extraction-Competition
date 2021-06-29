import torch
import numpy as np
import datetime


def sec_to_str(sec):
    time = str(datetime.timedelta(seconds=sec)).split(".")
    return time[0]


# 실험의 Randomness를 제거하여 실험이 같은 조건일 때 동일한 결과를 얻게 해줍니다.
def seed_everything(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    
    print(f'이 실험은 seed {seed}로 고정되었습니다.')
    
    
def use_cuda():
    if torch.cuda.is_available(): #checking for GPU availability
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    return device