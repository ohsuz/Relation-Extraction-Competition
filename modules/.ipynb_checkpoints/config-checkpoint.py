from .utils import *

train_dir = '/opt/ml/input/data/train'
test_dir = '/opt/ml/input/data/test'
model_dir = '/opt/ml/models'
submission_dir = '/opt/ml/submissions'
ensemble_dir = '/opt/ml/ensemble'
n_fold = 5

lr = 5e-5
batch_size = 64
num_workers = 5
num_epochs = 8
device = use_cuda()

MODEL_NAME = "monologg/koelectra-base-v3-discriminator"
TOK_NAME = "monologg/koelectra-base-v3-discriminator"
special_tokens_dict = {'additional_special_tokens': ['[E01]', '[/E01]', '[EO2]', '[/E02]']}
special_tokens_dict_2 = {'additional_special_tokens': ['#', '@', 'α', 'β']}