import json
import torch
from torch.utils.data import Dataset, DataLoader

from data.vocab import Vocab


class PhoNer(Dataset):
    def __init__(self, file_path: str, vocabulary: Vocab):
        super().__init__()

        self.file_path = file_path
        self.vocabulary = vocabulary

        self.dataset = json.load(open(file_path, 'r', encoding='utf-8'))
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        record = self.dataset[idx]

        token_list = record['words']
        tag_list = record['tags']

        encoded_tokens = self.vocabulary.encode_sentence(token_list)
        encoded_tags = self.vocabulary.encode_label(tag_list)

        return {
            "input_ids" : encoded_tokens,
            "label" : encoded_tags
        }
    

    @staticmethod
    def collate_fn(batch_samples: list[dict]) -> dict[dict]:
        batch_data = {
            "input_ids": torch.stack([sample['input_ids'] for sample in batch_samples], dim=0),
            'label': torch.stack([sample['label'] for sample in batch_samples], dim=0)
        }

        return batch_data
    

if __name__ == '__main__':
    file_path = r'PhoNER\test.json'
    phoner_dataset = PhoNer(file_path, Vocab(file_path, 'words', 'tags'))
    data_loader = DataLoader(phoner_dataset, 16, collate_fn=phoner_dataset.collate_fn)

    for batch_idx, batch_data in enumerate(data_loader):
        print(batch_data)
        raise