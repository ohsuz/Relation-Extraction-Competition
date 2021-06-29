from transformers import ElectraModel, ElectraTokenizer, ElectraForSequenceClassification
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
from transformers import AdamW
import torch
import argparse
from .loss import *


def get_model():
    model = ElectraForSequenceClassification.from_pretrained(args., num_labels=42).to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    criterion = CELSLoss()
    
    return model, optimizer, scheduler, criterion


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, default="monologg/koelectra-base-v3-discriminator")
    
    args = parser.parse_args()
    print(args)
    main(args)
