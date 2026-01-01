import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import os
import json


class Trainer:
    """Trainer cho các mô hình"""
    
    def __init__(self, model, train_loader, dev_loader, test_loader,
                 optimizer, scheduler=None, device='cuda', 
                 save_dir='checkpoints', task_type='classification'):
        """
        Args:
            model: PyTorch model
            train_loader, dev_loader, test_loader: DataLoaders
            optimizer: optimizer
            scheduler: learning rate scheduler
            device: 'cuda' hoặc 'cpu'
            save_dir: thư mục lưu checkpoints
            task_type: 'classification' hoặc 'token_classification'
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = save_dir
        self.task_type = task_type
        
        # Tạo thư mục lưu checkpoints
        os.makedirs(save_dir, exist_ok=True)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding index
        
        # Best metrics
        self.best_dev_metric = 0.0
        self.best_epoch = 0
        
        # History
        self.history = {
            'train_loss': [],
            'train_metric': [],
            'dev_loss': [],
            'dev_metric': []
        }
    
    def train_epoch(self):
        """Train một epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        progress_bar = tqdm(self.train_loader, desc='Training')
        for batch in progress_bar:
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask)
            
            # Compute loss
            if self.task_type == 'classification':
                loss = self.criterion(logits, labels)
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            else:  # token_classification
                # Flatten cho token classification
                loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                # Chỉ tính metrics cho non-padding tokens
                mask = labels != 0  # 0 là padding index
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds[mask].cpu().numpy())
                all_labels.extend(labels[mask].cpu().numpy())
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(self.train_loader)
        
        # Compute metrics
        if self.task_type == 'classification':
            metric = accuracy_score(all_labels, all_preds)
        else:  # token_classification
            _, _, metric, _ = precision_recall_fscore_support(
                all_labels, all_preds, average='macro', zero_division=0
            )
        
        return avg_loss, metric
    
    def evaluate(self, data_loader):
        """Evaluate trên một dataloader"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc='Evaluating'):
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                logits = self.model(input_ids, attention_mask)
                
                # Compute loss
                if self.task_type == 'classification':
                    loss = self.criterion(logits, labels)
                    preds = torch.argmax(logits, dim=-1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                else:  # token_classification
                    loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                    mask = labels != 0
                    preds = torch.argmax(logits, dim=-1)
                    all_preds.extend(preds[mask].cpu().numpy())
                    all_labels.extend(labels[mask].cpu().numpy())
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(data_loader)
        
        # Compute metrics
        if self.task_type == 'classification':
            metric = accuracy_score(all_labels, all_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_preds, average='macro', zero_division=0
            )
        else:  # token_classification
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_preds, average='macro', zero_division=0
            )
            metric = f1
        
        return avg_loss, metric, precision, recall, f1, all_preds, all_labels
    
    def train(self, num_epochs, patience=5):
        """
        Train model
        
        Args:
            num_epochs: số epochs
            patience: số epochs chờ đợi trước khi early stopping
        """
        print(f"Training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Task type: {self.task_type}")
        
        no_improve = 0
        
        for epoch in range(num_epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*50}")
            
            # Train
            train_loss, train_metric = self.train_epoch()
            
            # Evaluate on dev set
            dev_loss, dev_metric, dev_precision, dev_recall, dev_f1, _, _ = self.evaluate(self.dev_loader)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_metric'].append(train_metric)
            self.history['dev_loss'].append(dev_loss)
            self.history['dev_metric'].append(dev_metric)
            
            # Print metrics
            metric_name = 'Accuracy' if self.task_type == 'classification' else 'F1-Score'
            print(f"\nTrain Loss: {train_loss:.4f} | Train {metric_name}: {train_metric:.4f}")
            print(f"Dev Loss: {dev_loss:.4f} | Dev {metric_name}: {dev_metric:.4f}")
            print(f"Dev Precision: {dev_precision:.4f} | Dev Recall: {dev_recall:.4f} | Dev F1: {dev_f1:.4f}")
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(dev_metric)
                else:
                    self.scheduler.step()
                print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if dev_metric > self.best_dev_metric:
                self.best_dev_metric = dev_metric
                self.best_epoch = epoch + 1
                self.save_checkpoint('best_model.pt')
                print(f"✓ Saved best model with {metric_name}: {dev_metric:.4f}")
                no_improve = 0
            else:
                no_improve += 1
            
            # Early stopping
            if no_improve >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        print(f"\n{'='*50}")
        print(f"Training completed!")
        print(f"Best {metric_name}: {self.best_dev_metric:.4f} at epoch {self.best_epoch}")
        print(f"{'='*50}")
        
        # Save history
        self.save_history()
    
    def test(self):
        """Test model trên test set"""
        print(f"\n{'='*50}")
        print("Testing on test set...")
        print(f"{'='*50}")
        
        # Load best model
        self.load_checkpoint('best_model.pt')
        
        # Evaluate
        test_loss, test_metric, test_precision, test_recall, test_f1, preds, labels = self.evaluate(self.test_loader)
        
        # Print results
        metric_name = 'Accuracy' if self.task_type == 'classification' else 'F1-Score'
        print(f"\nTest Loss: {test_loss:.4f}")
        print(f"Test {metric_name}: {test_metric:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        print(f"Test F1: {test_f1:.4f}")
        
        # Classification report
        print("\nDetailed Classification Report:")
        print(classification_report(labels, preds, zero_division=0))
        
        return test_loss, test_metric, test_precision, test_recall, test_f1
    
    def save_checkpoint(self, filename):
        """Lưu checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_dev_metric': self.best_dev_metric,
            'best_epoch': self.best_epoch,
            'history': self.history
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        filepath = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filename):
        """Load checkpoint"""
        filepath = os.path.join(self.save_dir, filename)
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_dev_metric = checkpoint['best_dev_metric']
        self.best_epoch = checkpoint['best_epoch']
        self.history = checkpoint['history']
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Loaded checkpoint from {filepath}")
    
    def save_history(self):
        """Lưu training history"""
        filepath = os.path.join(self.save_dir, 'history.json')
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Saved history to {filepath}")


def create_trainer(model, train_loader, dev_loader, test_loader,
                   learning_rate=1e-4, device='cuda', save_dir='checkpoints',
                   task_type='classification'):
    """
    Tạo trainer với optimizer và scheduler
    
    Args:
        model: PyTorch model
        train_loader, dev_loader, test_loader: DataLoaders
        learning_rate: learning rate
        device: 'cuda' hoặc 'cpu'
        save_dir: thư mục lưu checkpoints
        task_type: 'classification' hoặc 'token_classification'
    
    Returns:
        trainer: Trainer object
    """
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='max',  # maximize metric
        factor=0.5, 
        patience=2,
        verbose=True
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=save_dir,
        task_type=task_type
    )
    
    return trainer
