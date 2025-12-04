import os

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.metrics import f1_score
from tqdm import tqdm

from utils.logging import setup_logger


class BaseTask:
    def __init__(self, vocabulary, neural_model, model_checkpoint_path):

        self.logger = setup_logger()

        self.model_checkpoint_path = model_checkpoint_path
        os.makedirs(self.model_checkpoint_path, exist_ok=True)

        self.logger.info("Creating vocab")
        self.vocabulary = vocabulary
        self.vocabulary.build_vocabulary()

        self.logger.info("Loading datasets & dataloaders")
        self.load_datasets()
        self.create_dataloaders()

        self.neural_model = neural_model

        self.compute_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.neural_model.to(self.compute_device)

        self.optimizer = Adam(self.neural_model.parameters(), lr=self.learning_rate)

    def load_datasets(self):
        raise NotImplementedError

    def create_dataloaders(self):
        raise NotImplementedError

    def forward_batch(self, batch):
        """Return loss, y_true, y_pred"""
        raise NotImplementedError


    def evaluate_metrics(self, data_loader, desc="Evaluating"):
        self.neural_model.eval()
        accumulated_loss = 0
        all_true_labels = []
        all_predictions = []

        progress_bar = tqdm(data_loader, desc=desc, ncols=90)

        with torch.no_grad():
            for batch_data in progress_bar:
                batch_data = {key: value.to(self.compute_device) for key, value in batch_data.items()}

                batch_loss, true_labels, predictions = self.forward_batch(batch_data)
                accumulated_loss += batch_loss.item()

                all_true_labels.extend(true_labels)
                all_predictions.extend(predictions)

                mean_loss = accumulated_loss / (progress_bar.n + 1)
                progress_bar.set_postfix(loss=f"{mean_loss:.4f}")

        mean_loss = accumulated_loss / len(data_loader)
        macro_f1_score = f1_score(all_true_labels, all_predictions, average="macro")

        return mean_loss, macro_f1_score


    def train(self, num_epochs=20, early_stop_patience=5):
        model_save_path = os.path.join(self.model_checkpoint_path, 'best_model.pt')

        best_f1_score = 0
        patience_count = 0

        for current_epoch in range(1, num_epochs + 1):

            self.neural_model.train()
            epoch_loss = 0

            progress_bar = tqdm(self.training_loader, desc=f"Epoch {current_epoch}/{num_epochs}", ncols=90)

            for batch_data in progress_bar:
                batch_data = {key: value.to(self.compute_device) for key, value in batch_data.items()}

                batch_loss, _, _ = self.forward_batch(batch_data)

                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

                epoch_loss += batch_loss.item()

                mean_loss = epoch_loss / (progress_bar.n + 1)
                progress_bar.set_postfix(loss=f"{mean_loss:.4f}")

            validation_loss, validation_f1 = self.evaluate_metrics(self.validation_loader, desc="Validating")

            self.logger.info(
                f"[Epoch {current_epoch}] Val_loss={validation_loss:.4f} | Val_Macro-F1={validation_f1:.4f}"
            )

            if validation_f1 > best_f1_score:
                best_f1_score = validation_f1
                patience_count = 0
                torch.save(self.neural_model.state_dict(), model_save_path)
                self.logger.info(f"New BEST model saved (F1={best_f1_score:.4f})")
            else:
                patience_count += 1
                if patience_count >= early_stop_patience:
                    self.logger.info("Early stopping triggered.")
                    break

        # Load best model
        self.neural_model.load_state_dict(torch.load(model_save_path, map_location=self.compute_device))


    def test(self):
        return self.evaluate_metrics(self.testing_loader, desc="Testing")

