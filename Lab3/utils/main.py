import argparse
import sys
sys.path.append('..')

from data.vocab import Vocab
from model.lstm import LSTM
from model.gru import GRU
from model.bi_lstm import BiLSTM
from tasks.text_classification import TextClassificationTask
from tasks.sequential_labeling import SequentialLabelingTask


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
    """Bài 3: BiLSTM cho nhận diện thực thể (PhoNER)"""
    print("\n" + "="*60)
    print("BÀI 3: BiLSTM cho nhận diện thực thể")
    print("="*60 + "\n")
    
    # Đường dẫn dữ liệu
    training_path = '../PhoNER/train.json'
    validation_path = '../PhoNER/dev.json'
    testing_path = '../PhoNER/test.json'
    
    # Tạo vocabulary
    vocabulary = Vocab(training_path, 'words', 'tags')
    
    # Khởi tạo mô hình BiLSTM
    bilstm_model = BiLSTM(vocabulary, embedding_size=256, hidden_size=256, layer_count=5)
    
    # Tạo task
    bilstm_task = SequentialLabelingTask(
        vocabulary=vocabulary,
        training_path=training_path,
        validation_path=validation_path,
        testing_path=testing_path,
        neural_model=bilstm_model,
        model_checkpoint_path='../checkpoints/bilstm',
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
