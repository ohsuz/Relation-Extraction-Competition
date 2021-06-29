from torch.utils.data import Dataset, DataLoader
from .config import *


class KoElecDataset(Dataset):
    def __init__(self, tsv_file, add_entity=True):
        self.dataset = load_data(tsv_file, add_entity)
        self.dataset['sentence'] = self.dataset['entity_01'] + ' [SEP] ' + self.dataset['entity_02'] + ' [SEP] ' + self.dataset['sentence']
        self.sentences = list(self.dataset['sentence'])
        self.labels = list(self.dataset['label'])
        self.tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sentence, label = self.sentences[idx], self.labels[idx]
        inputs = self.tokenizer(
            sentence,
            return_tensors='pt',
            truncation=True,
            max_length=190,
            pad_to_max_length=True,
            add_special_tokens=True
        )
        
        input_ids = inputs['input_ids'][0]
        attention_mask = inputs['attention_mask'][0]
        
        return input_ids, attention_mask, label
    

def load_data(dataset_dir, add_entity=True):
    with open('/opt/ml/input/data/label_type.pkl', 'rb') as f:
        label_type = pickle.load(f)
    dataset = pd.read_csv(dataset_dir, delimiter='\t', header=None)
    dataset = preprocessing_dataset(dataset, label_type, add_entity)
    return dataset


def preprocessing_dataset(dataset, label_type, add_entity):
    label = []
    sentences = None
    for i in dataset[8]:
        if i == 'blind':
            label.append(100)
        else:
            label.append(label_type[i])
    
    if add_entity:
        ### how to make this efficient ###
        sentences = [add_entity_tokens(dataset[1][i], dataset[3][i], dataset[4][i], dataset[6][i], dataset[7][i]) for i in range(len(dataset))]
    else:
        sentences = dataset[1]

    out_dataset = pd.DataFrame({'sentence':sentences,'entity_01':dataset[2],'entity_02':dataset[5],'label':label,})
    return out_dataset