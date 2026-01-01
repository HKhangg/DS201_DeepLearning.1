import json
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter


class NERVocabulary:
    """Vocabulary cho bài toán NER"""
    
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
        
    def build_vocab(self, sentences):
        """
        Xây dựng vocabulary từ list of sentences
        Args:
            sentences: list of list of words
        """
        # Đếm tần suất
        for sentence in sentences:
            self.word_freq.update(sentence)
        
        # Thêm các từ có tần suất >= min_freq
        idx = len(self.word2idx)
        for word, freq in self.word_freq.items():
            if freq >= self.min_freq and word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
    
    def encode(self, words):
        """Chuyển list of words thành list of indices"""
        return [self.word2idx.get(word, self.UNK_IDX) for word in words]
    
    def decode(self, indices):
        """Chuyển list of indices thành list of words"""
        return [self.idx2word.get(idx, self.UNK_TOKEN) for idx in indices]
    
    def __len__(self):
        return len(self.word2idx)


class LabelEncoder:
    """Encoder cho NER labels"""
    
    def __init__(self):
        self.label2idx = {}
        self.idx2label = {}
        
        # PAD label cho padding tokens
        self.PAD_LABEL = '<PAD>'
        self.PAD_IDX = 0
        
        self.label2idx[self.PAD_LABEL] = self.PAD_IDX
        self.idx2label[self.PAD_IDX] = self.PAD_LABEL
        
    def build_labels(self, tag_sequences):
        """
        Xây dựng label mapping từ tag sequences
        Args:
            tag_sequences: list of list of tags
        """
        unique_tags = set()
        for tags in tag_sequences:
            unique_tags.update(tags)
        
        # Loại bỏ PAD_LABEL nếu có trong data
        unique_tags.discard(self.PAD_LABEL)
        
        # Thêm các labels
        idx = len(self.label2idx)
        for tag in sorted(unique_tags):
            if tag not in self.label2idx:
                self.label2idx[tag] = idx
                self.idx2label[idx] = tag
                idx += 1
    
    def encode(self, tags):
        """Chuyển list of tags thành list of label indices"""
        return [self.label2idx.get(tag, self.PAD_IDX) for tag in tags]
    
    def decode(self, indices):
        """Chuyển list of label indices thành list of tags"""
        return [self.idx2label.get(idx, self.PAD_LABEL) for idx in indices]
    
    def __len__(self):
        return len(self.label2idx)


class PhoNERT_Dataset(Dataset):
    """Dataset cho bài toán NER - PhoNERT"""
    
    def __init__(self, data_path, vocab=None, label_encoder=None, max_len=128):
        """
        Args:
            data_path: đường dẫn đến file json
            vocab: NERVocabulary object
            label_encoder: LabelEncoder object
            max_len: độ dài tối đa của sequence
        """
        self.max_len = max_len
        
        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Build vocabulary nếu chưa có
        if vocab is None:
            self.vocab = NERVocabulary(min_freq=2)
            sentences = [item['words'] for item in self.data]
            self.vocab.build_vocab(sentences)
        else:
            self.vocab = vocab
        
        # Build label encoder nếu chưa có
        if label_encoder is None:
            self.label_encoder = LabelEncoder()
            tag_sequences = [item['tags'] for item in self.data]
            self.label_encoder.build_labels(tag_sequences)
        else:
            self.label_encoder = label_encoder
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Lấy words và tags
        words = item['words']
        tags = item['tags']
        
        # Truncate nếu quá dài
        if len(words) > self.max_len:
            words = words[:self.max_len]
            tags = tags[:self.max_len]
        
        # Encode
        input_ids = self.vocab.encode(words)
        labels = self.label_encoder.encode(tags)
        
        # Tạo attention mask
        attention_mask = [1] * len(input_ids)
        
        # Padding
        padding_length = self.max_len - len(input_ids)
        input_ids += [self.vocab.PAD_IDX] * padding_length
        attention_mask += [0] * padding_length
        labels += [self.label_encoder.PAD_IDX] * padding_length
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }


def create_phonert_dataloaders(train_path, dev_path, test_path, 
                                batch_size=32, max_len=128, num_workers=0):
    """
    Tạo dataloaders cho PhoNERT dataset
    
    Args:
        train_path: đường dẫn đến train file
        dev_path: đường dẫn đến dev file
        test_path: đường dẫn đến test file
        batch_size: batch size
        max_len: độ dài tối đa của sequence
        num_workers: số workers cho DataLoader
    
    Returns:
        train_loader, dev_loader, test_loader, vocab, label_encoder, num_labels
    """
    # Tạo train dataset và build vocabulary + label encoder
    train_dataset = PhoNERT_Dataset(train_path, vocab=None, label_encoder=None, max_len=max_len)
    vocab = train_dataset.vocab
    label_encoder = train_dataset.label_encoder
    num_labels = len(label_encoder)
    
    # Tạo dev và test dataset với vocab và label_encoder đã build
    dev_dataset = PhoNERT_Dataset(dev_path, vocab=vocab, label_encoder=label_encoder, max_len=max_len)
    test_dataset = PhoNERT_Dataset(test_path, vocab=vocab, label_encoder=label_encoder, max_len=max_len)
    
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
    
    return train_loader, dev_loader, test_loader, vocab, label_encoder, num_labels


if __name__ == '__main__':
    # Test dataset
    train_path = '../data/PhoNERT/train.json'
    dev_path = '../data/PhoNERT/dev.json'
    test_path = '../data/PhoNERT/test.json'
    
    train_loader, dev_loader, test_loader, vocab, label_encoder, num_labels = create_phonert_dataloaders(
        train_path, dev_path, test_path, batch_size=4
    )
    
    print(f"Vocab size: {len(vocab)}")
    print(f"Number of labels: {num_labels}")
    print(f"Label mapping: {label_encoder.label2idx}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Dev batches: {len(dev_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test một batch
    for batch in train_loader:
        print("\nSample batch:")
        print(f"Input IDs shape: {batch['input_ids'].shape}")
        print(f"Attention mask shape: {batch['attention_mask'].shape}")
        print(f"Labels shape: {batch['labels'].shape}")
        
        # Decode sample
        print("\nFirst sample in batch:")
        words = vocab.decode(batch['input_ids'][0].tolist())
        tags = label_encoder.decode(batch['labels'][0].tolist())
        print(f"Words: {words[:10]}")
        print(f"Tags: {tags[:10]}")
        break
