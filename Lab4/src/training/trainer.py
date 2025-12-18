"""
Training pipeline for neural machine translation models.

This module provides a flexible training framework for seq2seq models
with support for early stopping, checkpointing, and ROUGE evaluation.
"""

import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from typing import Optional, Tuple
from tqdm import tqdm
from rouge import Rouge

from src.utils.logging import setup_logger
from src.data.vocab import Vocab
from src.data.PhoMT import PhoMT


class Trainer:
    """
    Trainer for sequence-to-sequence translation models.
    
    Handles training loop, validation, testing, and checkpointing
    with support for early stopping based on ROUGE-L scores.
    """
    
    def __init__(
        self,
        vocab: Vocab,
        model: nn.Module,
        train_path: str,
        dev_path: Optional[str] = None,
        test_path: Optional[str] = None,
        logger=None,
        checkpoint_path: str = "checkpoints",
        learning_rate: float = 1e-3,
        batch_size: int = 32
    ):
        """
        Initialize the trainer.
        
        Args:
            vocab: Vocabulary object for source and target languages
            model: Seq2Seq model to train
            train_path: Path to training data
            dev_path: Path to validation data
            test_path: Path to test data
            logger: Logger instance (created if None)
            checkpoint_path: Directory to save model checkpoints
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
        """
        # Setup logging
        self.logger = logger if logger is not None else setup_logger(
            output=checkpoint_path
        )
        
        # Setup checkpoint directory
        self.checkpoint_path = checkpoint_path
        os.makedirs(self.checkpoint_path, exist_ok=True)
        
        # Store vocabulary
        self.logger.info("Initializing vocabulary")
        self.vocab = vocab
        
        # Load datasets
        self.logger.info("Loading datasets")
        self.train_dataset = PhoMT(train_path, self.vocab)
        self.val_dataset = PhoMT(dev_path, self.vocab) if dev_path else None
        self.test_dataset = PhoMT(test_path, self.vocab) if test_path else None
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.train_dataset.collate_fn
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            collate_fn=self.val_dataset.collate_fn
        ) if self.val_dataset else None
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            collate_fn=self.test_dataset.collate_fn
        ) if self.test_dataset else None
        
        # Setup device and model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.logger.info(f"Using device: {self.device}")
        
        # Setup optimizer and loss
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.vocab.pad_id)
        
        # Setup evaluation metric
        self.rouge_scorer = Rouge()
    
    def process_batch(self, batch) -> Tuple[torch.Tensor, list, list]:
        """
        Process a single batch through the model.
        
        Args:
            batch: Dictionary containing 'src_ids' and 'tar_ids'
            
        Returns:
            loss: Loss value for the batch
            references: List of reference translations
            predictions: List of model predictions
        """
        src_ids = batch["src_ids"].to(self.device)
        tar_ids = batch["tar_ids"].to(self.device)
        
        # Forward pass
        loss = self.model(src_ids, tar_ids)
        
        # Generate predictions during evaluation
        if not self.model.training:
            pred_ids = self.model.predict(src_ids)
        else:
            pred_ids = None
        
        # Decode sequences to text
        references = []
        predictions = []
        
        for i in range(src_ids.size(0)):
            # Decode reference
            ref = self.vocab.decode_sentence(
                tar_ids[i].tolist(), 
                self.vocab.tar_lang
            )
            references.append(ref)
            
            # Decode prediction
            if pred_ids is not None:
                pred = self.vocab.decode_sentence(
                    pred_ids[i].tolist(), 
                    self.vocab.tar_lang
                )
                predictions.append(pred)
            else:
                predictions.append("")
        
        return loss, references, predictions
    
    def evaluate(
        self, 
        dataloader: DataLoader, 
        description: str = "Evaluating"
    ) -> Tuple[float, float]:
        """
        Evaluate model on a dataset.
        
        Args:
            dataloader: DataLoader for evaluation
            description: Description for progress bar
            
        Returns:
            average_loss: Average loss over the dataset
            average_rouge: Average ROUGE-L F1 score
        """
        self.model.eval()
        total_loss = 0.0
        rouge_scores = []
        
        progress_bar = tqdm(dataloader, desc=description, ncols=100)
        
        with torch.no_grad():
            for batch in progress_bar:
                # Process batch
                loss, references, predictions = self.process_batch(batch)
                total_loss += loss.item()
                
                # Compute ROUGE scores
                batch_scores = self.rouge_scorer.get_scores(
                    predictions, 
                    references, 
                    avg=True
                )
                rouge_l_f1 = batch_scores["rouge-l"]["f"]
                rouge_scores.append(rouge_l_f1)
                
                # Update progress bar
                current_loss = total_loss / (progress_bar.n + 1)
                current_rouge = sum(rouge_scores) / len(rouge_scores)
                
                progress_bar.set_postfix(
                    loss=f"{current_loss:.4f}",
                    rouge_l=f"{current_rouge:.4f}"
                )
        
        average_loss = total_loss / len(dataloader)
        average_rouge = sum(rouge_scores) / len(rouge_scores)
        
        return average_loss, average_rouge
    
    def train(self, num_epochs: int = 20, patience: int = 5):
        """
        Train the model with early stopping.
        
        Args:
            num_epochs: Maximum number of training epochs
            patience: Number of epochs to wait for improvement before stopping
        """
        best_rouge_score = 0.0
        epochs_without_improvement = 0
        
        # Define checkpoint path
        model_name = self.model.__class__.__name__
        checkpoint_file = os.path.join(
            self.checkpoint_path,
            f"best_{model_name}.pt"
        )
        
        self.logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(1, num_epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            
            # Training loop
            progress_bar = tqdm(
                self.train_loader, 
                desc=f"Epoch {epoch}/{num_epochs}", 
                ncols=100
            )
            
            for batch in progress_bar:
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                loss, _, _ = self.process_batch(batch)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Update metrics
                epoch_loss += loss.item()
                current_loss = epoch_loss / (progress_bar.n + 1)
                
                progress_bar.set_postfix(loss=f"{current_loss:.4f}")
            
            # Validation
            if self.val_loader is not None:
                val_loss, val_rouge = self.evaluate(
                    self.val_loader, 
                    description="Validating"
                )
                
                self.logger.info(
                    f"[Epoch {epoch}/{num_epochs}] "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val ROUGE-L: {val_rouge:.4f}"
                )
                
                # Save best model
                if val_rouge > best_rouge_score:
                    best_rouge_score = val_rouge
                    epochs_without_improvement = 0
                    
                    torch.save(self.model.state_dict(), checkpoint_file)
                    self.logger.info(
                        f"New best model saved! "
                        f"ROUGE-L: {best_rouge_score:.4f}"
                    )
                else:
                    epochs_without_improvement += 1
                    self.logger.info(
                        f"No improvement for {epochs_without_improvement} epoch(s)"
                    )
                    
                    # Early stopping
                    if epochs_without_improvement >= patience:
                        self.logger.info(
                            f"Early stopping triggered after {epoch} epochs"
                        )
                        break
        
        # Load best model
        self.logger.info("Loading best model checkpoint")
        self.model.load_state_dict(
            torch.load(checkpoint_file, map_location=self.device)
        )
    
    def test(self) -> Tuple[float, float]:
        """
        Evaluate model on test set.
        
        Returns:
            test_loss: Test loss
            test_rouge: Test ROUGE-L F1 score
        """
        if self.test_loader is None:
            raise ValueError("Test dataset not provided")
        
        self.logger.info("Evaluating on test set")
        return self.evaluate(self.test_loader, description="Testing")
