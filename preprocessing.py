import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torchvision import transforms
from pathlib import Path
import numpy as np
from glob import glob


class Process:

    def __init__(self) -> None:
        pass

    def __call__(self, path, test=False):
        assert path.split('/')[-1] == 'train' or path.split('/')[-1] == 'test'
        print('Loading sentences...')
        sentences = list()
        labels = list()

        for path in glob(f'./aclImdb/train/pos/*.txt') + glob(f'./aclImdb/train/neg/*.txt'):
            with open(path, 'r', encoding='utf-8') as f:
                sentences.append(f.read())
                labels.append(1 if 'pos' in path else 0)
                
        if test:
            assert len(np.unique(labels))==2
            assert min(labels) == 0
            assert max(labels) == 1
        print(f'Loaded {len(sentences)} sentences.')
        return sentences, labels


class Data(Dataset):
    
    def __init__(self, X, y, transform=None) -> None:
        """Used together with class Dataloader to structure and load dataset."""

        self.X, self.y = X, y
        self.transform  = transform
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'X': self.X[idx], 'y': self.y[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Tokenize:

    MAX_SEQ_LEN = 512  # Max sequence length allowed for BERT.

    def __init__(self) -> None:
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    def __call__(self, sample):
        text = sample['X'] if isinstance(sample, dict) else sample
        tokenized_text = self.tokenizer.tokenize(text)
        if len(tokenized_text) > Tokenize.MAX_SEQ_LEN - 2:
            tokenized_text = self._truncate(tokenized_text)
        encoded_sequence = self.tokenizer.encode_plus(tokenized_text, return_tensors='pt', 
                                                   padding='max_length', max_length=512,
                                                   add_special_tokens=True, return_attention_mask=True)
        if isinstance(sample, dict):
            return {'X': encoded_sequence.to('cuda'), 'y': sample['y']}
        else:
            return encoded_sequence

    def _truncate(self, sequence):
        """Truncates input sequence above allowed length.
        
        Keeps the first 128 and the last 382 tokens.
        Source: https://arxiv.org/pdf/1905.05583.pdf"""

        return sequence[:128] + sequence[-382:]
