import torch
from pyvi import ViTokenizer
import json

class Vocab():
    def __init__(self, data_path, text_field: str, target_field:str, sequence_length: int = 100):
        self.data_path = data_path
        self.text_field = text_field
        self.target_field = target_field
        self.sequence_length = sequence_length

        self.vocabulary = [] 
        self.target_classes = set()

        self.padding_idx = 0
        self.unknown_idx = 1

        self.token_to_idx = {}
        self.idx_to_token = {}

        self.idx_to_label = {}
        self.label_to_idx = {}

        self.build_vocabulary()

    def build_vocabulary(self):
        # Đọc dataset - hỗ trợ cả JSON array và JSON Lines format
        dataset = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        # Thử JSON array trước
        try:
            dataset = json.loads(content)
        except json.JSONDecodeError:
            # Nếu lỗi, thử JSON Lines (mỗi dòng là 1 object)
            for line in content.split('\n'):
                line = line.strip()
                if line:
                    dataset.append(json.loads(line))

        for record in dataset:

            try:
                # Text classification
                if isinstance(record[self.text_field], str):
                    processed_text = ViTokenizer.tokenize(record[self.text_field])
                    self.vocabulary.extend(processed_text.split())
                
                if isinstance(record[self.target_field], str):
                    self.target_classes.add(record[self.target_field])


                # Sequential labeling
                if isinstance(record[self.text_field], list):
                    for token in record[self.text_field]:
                        self.vocabulary.append(token)
                
                if isinstance(record[self.target_field], list):
                    for tag in record[self.target_field]:
                        self.target_classes.add(tag)

            except:
                raise Exception('Wrong input type')

        self.vocabulary = list(set(self.vocabulary))

        self.token_to_idx = {token : index for index, token in enumerate(self.vocabulary, 2)}
        self.idx_to_token = {index : token for token, index in self.token_to_idx.items()}

        self.label_to_idx = {label : index for index, label in enumerate(self.target_classes)}
        self.idx_to_label = {index: label for label, index in self.label_to_idx.items()}


    def encode_sentence(self, text_input: str):
        try:
            if isinstance(text_input, str): # Text classification
                processed_text = ViTokenizer.tokenize(text_input)
                token_list = processed_text.split()

            if isinstance(text_input, list): # Sequential labeling
                token_list = text_input
        except:
            raise Exception('Wrong input type')


        encoded_tokens = []
        for token in token_list:
            try:
                encoded_tokens.append(self.token_to_idx[token])
            except:
                encoded_tokens.append(self.unknown_idx)


        if len(encoded_tokens) > self.sequence_length:
            encoded_tokens = encoded_tokens[:self.sequence_length] # Truncation
        else:
            encoded_tokens.extend([self.padding_idx] * (self.sequence_length - len(encoded_tokens))) # Padding

        return torch.tensor(encoded_tokens, dtype=torch.long)
    
        
    def encode_label(self, target_labels: str):
        if isinstance(target_labels, str): # Text classification
            return torch.tensor([self.label_to_idx[target_labels]], dtype=torch.long)
        
        elif isinstance(target_labels, list): # Sequential labeling
            encoded_labels = [self.label_to_idx[label] for label in target_labels]
            
            if len(target_labels) > self.sequence_length:
                encoded_labels = encoded_labels[:self.sequence_length]
                
            else:
                encoded_labels.extend([-100] * (self.sequence_length - len(target_labels)))


            return torch.tensor(encoded_labels, dtype=torch.long)
        
        else:
            raise Exception('Wrong input type')


    def decode_label(self, encoded_label_vector: torch.Tensor):
        decoded_labels = []
        for label_index in encoded_label_vector:
            index_value = label_index.item()
            if index_value == -100: # ignore index for cross entropy loss
                continue
            decoded_labels.append(self.idx_to_label[index_value])
        return decoded_labels
    
    @property
    def vocab_size(self):
        return len(self.vocabulary) + 2 # unk, pad id

    @property
    def num_labels(self):
        return len(self.target_classes)

if __name__ == '__main__':
    # file_path = r'PhoNER\word\test_word.json'
    # records = []
    # with open(file_path, 'r', encoding='utf-8') as file:
    #     for line in file:
    #         record = json.loads(line)
    #         records.append(record)

    # with open("PhoNER/test.json", "w", encoding="utf-8") as f:
    #     json.dump(records, f, ensure_ascii=False, indent=4)

    file_path = r'UIT-VSFC\UIT-VSFC-train.json'
    text_field='sentence' 
    target_field='sentiment'

    vocab_instance = Vocab(file_path, text_field, target_field)

    test_sample = {
        'sentence' : 'giảng buồn ngủ',
        'label': 'negative'
    }

    print(vocab_instance.encode_sentence(test_sample['sentence']))
    print(vocab_instance.vocab_size)
    print(vocab_instance.num_labels)
    print(vocab_instance.label_to_idx)
    print(vocab_instance.encode_label(test_sample['label']))
    print(vocab_instance.decode_label(vocab_instance.encode_label(test_sample['label'])))

    test_sample = {
        "words": [
            "Từ",
            "24",
            "-",
            "7",
            "đến",
            "31",
            "-",
            "7",
            ",",
            "bệnh",
            "nhân",
            "được",
            "mẹ",
            "là",
            "bà",
            "H.T.P",
            "(",
            "47",
            "tuổi",
            ")",
            "đón",
            "về",
            "nhà",
            "ở",
            "phường",
            "Phước",
            "Hoà",
            "(",
            "bằng",
            "xe",
            "máy",
            ")",
            ",",
            "không",
            "đi",
            "đâu",
            "chỉ",
            "ra",
            "Tạp",
            "hoá",
            "Phượng",
            ",",
            "chợ",
            "Vườn",
            "Lài",
            ",",
            "phường",
            "An",
            "Sơn",
            "cùng",
            "mẹ",
            "bán",
            "tạp",
            "hoá",
            "ở",
            "đây",
            "."
        ],
        "tags": [
            "O",
            "B-DATE",
            "I-DATE",
            "I-DATE",
            "O",
            "B-DATE",
            "I-DATE",
            "I-DATE",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "B-NAME",
            "O",
            "B-AGE",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "B-LOCATION",
            "I-LOCATION",
            "I-LOCATION",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "B-LOCATION",
            "I-LOCATION",
            "I-LOCATION",
            "O",
            "B-LOCATION",
            "I-LOCATION",
            "I-LOCATION",
            "O",
            "B-LOCATION",
            "I-LOCATION",
            "I-LOCATION",
            "O",
            "O",
            "B-JOB",
            "I-JOB",
            "I-JOB",
            "O",
            "O",
            "O"
        ]
    }
    file_path = 'PhoNER\dev.json'
    text_field='words' 
    target_field='tags'

    vocab_instance = Vocab(file_path, text_field, target_field)
    print(vocab_instance.vocab_size)
    print(vocab_instance.num_labels)
    print(vocab_instance.label_to_idx)


    print(vocab_instance.encode_sentence(test_sample['words']))

    encoded_target = vocab_instance.encode_label(test_sample['tags'])
    print(encoded_target)
    print(vocab_instance.decode_label(encoded_target))