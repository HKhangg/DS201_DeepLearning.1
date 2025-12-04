import json
import torch
from torch.utils.data import Dataset, DataLoader

from data.vocab import Vocab


class UIT_VSFC(Dataset):
    def __init__(self, file_path: str, target_field: str, vocabulary: Vocab):
        super().__init__()

        self.file_path = file_path
        self.target_field = target_field
        self.vocabulary = vocabulary

        self.dataset = json.load(open(file_path, 'r', encoding='utf-8'))
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        record = self.dataset[idx]

        text_content = record['sentence']
        sentiment_label = record[self.target_field]

        encoded_text = self.vocabulary.encode_sentence(text_content)
        encoded_sentiment = self.vocabulary.encode_label(sentiment_label)

        return {
            "input_ids" : encoded_text,
            "label" : encoded_sentiment
        }
    

    @staticmethod
    def collate_fn(batch_samples: list[dict]) -> dict[dict]:
        batch_data = {
            "input_ids": torch.stack([sample['input_ids'] for sample in batch_samples], dim=0),
            'label': torch.stack([sample['label'] for sample in batch_samples], dim=0)
        }

        return batch_data
    

if __name__ == '__main__':
    file_path = r'UIT-VSFC\\UIT-VSFC-train.json'
    vsfc_dataset = UIT_VSFC(file_path, 'sentiment', Vocab(file_path, 'sentence', 'sentiment'))
    data_loader = DataLoader(vsfc_dataset, 16, collate_fn=vsfc_dataset.collate_fn)

    for batch_idx, batch_data in enumerate(data_loader):
        print(batch_data)
        raise
        raise