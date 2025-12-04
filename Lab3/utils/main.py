import argparse
import sys
import os
import urllib.request
from pathlib import Path

sys.path.append('..')

from data.vocab import Vocab
from model.lstm import LSTM
from model.gru import GRU
from model.bi_lstm import BiLSTM
from tasks.text_classification import TextClassificationTask
from tasks.sequential_labeling import SequentialLabelingTask


def download_phoner_dataset():
    """Tự động tải dataset PhoNER về"""
    print("Đang tải PhoNER dataset...")
    
    # Tạo thư mục
    data_dir = Path('../data/phoner/word')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    base_url = 'https://raw.githubusercontent.com/VinAIResearch/PhoNER_COVID19/main/data/word/'
    files = ['train_word.json', 'dev_word.json', 'test_word.json']
    
    for file in files:
        file_path = data_dir / file
        
        # Skip nếu file đã tồn tại
        if file_path.exists():
            print(f"✓ {file} đã tồn tại")
            continue
            
        try:
            url = base_url + file
            urllib.request.urlretrieve(url, str(file_path))
            print(f"✓ Đã tải {file}")
        except Exception as e:
            print(f"✗ Lỗi khi tải {file}: {e}")
            return False
    
    print("✓ Hoàn thành tải dataset!\n")
    return True


def run_task1():
    """Bài 1: LSTM cho phân loại văn bản (UIT-VSFC)"""
    print("\n" + "="*60)
    print("BÀI 1: LSTM cho phân loại văn bản")
    print("="*60 + "\n")
    
    # Đường dẫn dữ liệu
    training_path = '/kaggle/input/uit-vsfc/UIT-VSFC-train.json'
    validation_path = '/kaggle/input/uit-vsfc/UIT-VSFC-dev.json'
    testing_path = '/kaggle/input/uit-vsfc/UIT-VSFC-test.json'
    
    # Tạo vocabulary
    vocabulary = Vocab(training_path, 'sentence', 'sentiment')
    
    # Khởi tạo mô hình LSTM
    lstm_model = LSTM(vocabulary, embedding_size=256, hidden_size=256, layer_count=5)
    
    # Tạo task
    lstm_task = TextClassificationTask(
        vocabulary=vocabulary,
        training_path=training_path,
        validation_path=validation_path,
        testing_path=testing_path,
        neural_model=lstm_model,
        model_checkpoint_path='../checkpoints/lstm',
        learning_rate=1e-3
    )
    
    # Huấn luyện
    lstm_task.train(num_epochs=20, early_stop_patience=5)
    
    # Đánh giá
    test_loss, test_f1 = lstm_task.test()
    print(f"\n{'='*60}")
    print(f"KẾT QUẢ LSTM:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Macro F1-Score: {test_f1:.4f}")
    print(f"{'='*60}\n")


def run_task2():
    """Bài 2: GRU cho phân loại văn bản (UIT-VSFC)"""
    print("\n" + "="*60)
    print("BÀI 2: GRU cho phân loại văn bản")
    print("="*60 + "\n")
    
    # Đường dẫn dữ liệu
    training_path = '/kaggle/input/uit-vsfc/UIT-VSFC-train.json'
    validation_path = '/kaggle/input/uit-vsfc/UIT-VSFC-dev.json'
    testing_path = '/kaggle/input/uit-vsfc/UIT-VSFC-test.json'
    
    # Tạo vocabulary
    vocabulary = Vocab(training_path, 'sentence', 'sentiment')
    
    # Khởi tạo mô hình GRU
    gru_model = GRU(vocabulary, embedding_size=256, hidden_size=256, layer_count=5)
    
    # Tạo task
    gru_task = TextClassificationTask(
        vocabulary=vocabulary,
        training_path=training_path,
        validation_path=validation_path,
        testing_path=testing_path,
        neural_model=gru_model,
        model_checkpoint_path='../checkpoints/gru',
        learning_rate=1e-3
    )
    
    # Huấn luyện
    gru_task.train(num_epochs=20, early_stop_patience=5)
    
    # Đánh giá
    test_loss, test_f1 = gru_task.test()
    print(f"\n{'='*60}")
    print(f"KẾT QUẢ GRU:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Macro F1-Score: {test_f1:.4f}")
    print(f"{'='*60}\n")


def run_task3():
    """Bài 3: Bi-LSTM cho gán nhãn chuỗi (Phoner)"""
    print("\n" + "="*60)
    print("BÀI 3: Bi-LSTM cho gán nhãn chuỗi")
    print("="*60 + "\n")
    
    # Tự động tải dataset nếu chưa có
    if not download_phoner_dataset():
        print("Lỗi: Không thể tải dataset. Vui lòng tải thủ công.")
        return
    
    # Đường dẫn dữ liệu
    base_path = '../data/phoner/word'
    training_path = f'{base_path}/train_word.json'
    validation_path = f'{base_path}/dev_word.json'
    testing_path = f'{base_path}/test_word.json'
    
    # Tạo vocabulary
    vocabulary = Vocab(training_path, 'words', 'tags')
    
    # Khởi tạo mô hình BiLSTM
    bilstm_model = BiLSTM(vocabulary, embedding_size=128, hidden_size=256, layer_count=2)
    
    # Tạo task
    checkpoint_dir = '../checkpoints/bilstm'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    bilstm_task = SequentialLabelingTask(
        vocabulary=vocabulary,
        training_path=training_path,
        validation_path=validation_path,
        testing_path=testing_path,
        neural_model=bilstm_model,
        model_checkpoint_path=checkpoint_dir,
        learning_rate=1e-3
    )
    
    # Huấn luyện
    bilstm_task.train(num_epochs=20, early_stop_patience=5)
    
    # Đánh giá
    test_loss, test_f1 = bilstm_task.test()
    print(f"\n{'='*60}")
    print(f"KẾT QUẢ BiLSTM:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Macro F1-Score: {test_f1:.4f}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Chạy các bài thực hành RNN')
    parser.add_argument('--task', type=int, required=True, choices=[1, 2, 3],
                        help='Số thứ tự bài tập (1, 2, hoặc 3)')
    
    args = parser.parse_args()
    
    if args.task == 1:
        run_task1()
    elif args.task == 2:
        run_task2()
    elif args.task == 3:
        run_task3()


if __name__ == '__main__':
    main()
