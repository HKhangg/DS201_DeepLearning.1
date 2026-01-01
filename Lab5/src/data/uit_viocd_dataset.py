import json
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re


class Vocabulary:
    """Xây dựng vocabulary từ dữ liệu"""
    
    def __init__(self, min_freq=1):
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}
        self.word_freq = Counter()
        
        # Special tokens
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        self.PAD_IDX = 0
        self.UNK_IDX = 1
        
        self.word2idx[self.PAD_TOKEN] = self.PAD_IDX
        self.word2idx[self.UNK_TOKEN] = self.UNK_IDX
        self.idx2word[self.PAD_IDX] = self.PAD_TOKEN
        self.idx2word[self.UNK_IDX] = self.UNK_TOKEN
        
    def build_vocab(self, texts):
        """Xây dựng vocabulary từ list of texts"""
        # Đếm tần suất
        for text in texts:
            tokens = self.tokenize(text)
            self.word_freq.update(tokens)
        
        # Thêm các từ có tần suất >= min_freq
        idx = len(self.word2idx)
        for word, freq in self.word_freq.items():
            if freq >= self.min_freq and word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
    
    def tokenize(self, text):
        """Tokenize text - simple word-level tokenization"""
        # Lowercase và tách từ
        text = text.lower()
        tokens = text.split()
        return tokens
    
    def encode(self, text):
        """Chuyển text thành list of indices"""
        tokens = self.tokenize(text)
        return [self.word2idx.get(token, self.UNK_IDX) for token in tokens]
    
    def decode(self, indices):
        """Chuyển list of indices thành text"""
        return ' '.join([self.idx2word.get(idx, self.UNK_TOKEN) for idx in indices])
    
    def __len__(self):
        return len(self.word2idx)


class UIT_ViOCD_Dataset(Dataset):
    """Dataset cho bài toán phân loại domain - UIT-ViOCD"""
    
    def __init__(self, data_path, vocab=None, max_len=128):
        """
        Args:
            data_path: đường dẫn đến file json
            vocab: Vocabulary object, nếu None sẽ tạo mới
            max_len: độ dài tối đa của sequence
        """
        self.max_len = max_len
        
        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Tạo label mapping
        self.domain2idx = self._create_label_mapping()
        self.idx2domain = {idx: domain for domain, idx in self.domain2idx.items()}
        
        # Build vocabulary nếu chưa có
        if vocab is None:
            self.vocab = Vocabulary(min_freq=2)
            texts = [item['review'] for item in self.data]
            self.vocab.build_vocab(texts)
        else:
            self.vocab = vocab
    
    def _create_label_mapping(self):
        """Tạo mapping từ domain label sang index"""
        domains = set([item['domain'] for item in self.data])
        return {domain: idx for idx, domain in enumerate(sorted(domains))}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Encode text
        text = item['review']
        input_ids = self.vocab.encode(text)
        
        # Truncate hoặc pad
        if len(input_ids) > self.max_len:
            input_ids = input_ids[:self.max_len]
        
        # Tạo attention mask (1 cho token thật, 0 cho padding)
        attention_mask = [1] * len(input_ids)
        
        # Padding
        padding_length = self.max_len - len(input_ids)
        input_ids += [self.vocab.PAD_IDX] * padding_length
        attention_mask += [0] * padding_length
        
        # Label
        label = self.domain2idx[item['domain']]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def create_uit_viocd_dataloaders(train_path, dev_path, test_path, 
                                  batch_size=32, max_len=128, num_workers=0):
    """
    Tạo dataloaders cho UIT-ViOCD dataset
    
    Args:
        train_path: đường dẫn đến train file
        dev_path: đường dẫn đến dev file  
        test_path: đường dẫn đến test file
        batch_size: batch size
        max_len: độ dài tối đa của sequence
        num_workers: số workers cho DataLoader
    
    Returns:
        train_loader, dev_loader, test_loader, vocab, num_classes
    """
    # Tạo train dataset và build vocabulary
    train_dataset = UIT_ViOCD_Dataset(train_path, vocab=None, max_len=max_len)
    vocab = train_dataset.vocab
    num_classes = len(train_dataset.domain2idx)
    
    # Tạo dev và test dataset với vocab đã build
    dev_dataset = UIT_ViOCD_Dataset(dev_path, vocab=vocab, max_len=max_len)
    test_dataset = UIT_ViOCD_Dataset(test_path, vocab=vocab, max_len=max_len)
    
    # Tạo dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, dev_loader, test_loader, vocab, num_classes


if __name__ == '__main__':
    # Test dataset
    train_path = '../data/UIT_ViOCD/train_preprocessed.json'
    dev_path = '../data/UIT_ViOCD/dev_preprocessed.json'
    test_path = '../data/UIT_ViOCD/test_preprocessed.json'
    
    train_loader, dev_loader, test_loader, vocab, num_classes = create_uit_viocd_dataloaders(
        train_path, dev_path, test_path, batch_size=4
    )
    
    print(f"Vocab size: {len(vocab)}")
    print(f"Number of classes: {num_classes}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Dev batches: {len(dev_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test một batch
    for batch in train_loader:
        print("\nSample batch:")
        print(f"Input IDs shape: {batch['input_ids'].shape}")
        print(f"Attention mask shape: {batch['attention_mask'].shape}")
        print(f"Labels shape: {batch['labels'].shape}")
        print(f"Labels: {batch['labels']}")
        break
