import torch
from torch.utils.data import Dataset, DataLoader
import json
from typing import Dict, List

from src.data.vocab import Vocab


class PhoMTDataset(Dataset):
    def __init__(self, json_path: str, vocabulary: Vocab):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)
        self.vocab = vocabulary
        print(f"Loaded {len(self.samples)} samples")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        src_text = sample[self.vocab.src_lang]
        tar_text = sample[self.vocab.tar_lang]
        
        return {
            "src_ids": self.vocab.encode(src_text, self.vocab.src_lang),
            "tar_ids": self.vocab.encode(tar_text, self.vocab.tar_lang)
        }
    
    @staticmethod
    def collate_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        return {
            "src_ids": torch.stack([item['src_ids'] for item in batch]),
            "tar_ids": torch.stack([item['tar_ids'] for item in batch])
        }


def build_dataloader(dataset: PhoMTDataset, batch_size: int = 32, 
                     shuffle: bool = True) -> DataLoader:
    return DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=PhoMTDataset.collate_batch,
        pin_memory=True
    )


if __name__ == '__main__':
    import numpy as np
    
    train_file = r'src\data\small-train.json'
    
    vocab = Vocab(train_file, 100, 'vietnamese', 'english')
    dataset = PhoMTDataset(train_file, vocab)
    loader = build_dataloader(dataset, batch_size=16)
    
    print(f"\nDataset size: {len(dataset)}")
    print(f"Number of batches: {len(loader)}")
    
    batch = next(iter(loader))
    print(f"\nBatch shapes:")
    print(f"  src_ids: {batch['src_ids'].shape}")
    print(f"  tar_ids: {batch['tar_ids'].shape}")
    
    with open(train_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    en_lens = [len(item['english'].split()) for item in data]
    vi_lens = [len(item['vietnamese'].split()) for item in data]
    
    print(f"\nEnglish - min: {np.min(en_lens)}, max: {np.max(en_lens)}, "
          f"mean: {np.mean(en_lens):.1f}, 95%: {np.percentile(en_lens, 95):.0f}")
    print(f"Vietnamese - min: {np.min(vi_lens)}, max: {np.max(vi_lens)}, "
          f"mean: {np.mean(vi_lens):.1f}, 95%: {np.percentile(vi_lens, 95):.0f}")
