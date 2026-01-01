import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """Set random seed cho reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    """Đếm số parameters của model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds):
    """Format thời gian từ seconds"""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def save_predictions(predictions, labels, output_file):
    """Lưu predictions ra file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for pred, label in zip(predictions, labels):
            f.write(f"{pred}\t{label}\n")


def load_config(config_file):
    """Load config từ file"""
    import json
    with open(config_file, 'r') as f:
        return json.load(f)


def get_device():
    """Lấy device (cuda hoặc cpu)"""
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = 'cpu'
        print("Using CPU")
    return device


def print_model_info(model):
    """In thông tin về model"""
    print("\n" + "="*50)
    print("MODEL INFORMATION")
    print("="*50)
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Model size: {count_parameters(model) * 4 / 1024 / 1024:.2f} MB")
    print("="*50 + "\n")
