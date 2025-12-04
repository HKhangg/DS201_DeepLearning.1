import torch
from torch.utils.data import DataLoader

from data.phoner import PhoNer
from tasks.base_task import BaseTask


class SequentialLabelingTask(BaseTask):

    def __init__(self, vocabulary, training_path, validation_path, testing_path,
                 neural_model, model_checkpoint_path, learning_rate=1e-3):

        self.training_path = training_path
        self.validation_path = validation_path
        self.testing_path = testing_path
        self.neural_model = neural_model
        self.model_checkpoint_path = model_checkpoint_path
        self.learning_rate = learning_rate

        super().__init__(vocabulary, neural_model, model_checkpoint_path)

    def load_datasets(self):
        self.training_dataset = PhoNer(self.training_path, self.vocabulary)
        self.validation_dataset = PhoNer(self.validation_path, self.vocabulary)
        self.testing_dataset = PhoNer(self.testing_path, self.vocabulary)

    def create_dataloaders(self):
        self.training_loader = DataLoader(self.training_dataset, batch_size=32, shuffle=True,
                                       collate_fn=self.training_dataset.collate_fn)
        self.validation_loader = DataLoader(self.validation_dataset, batch_size=32,
                                     collate_fn=self.validation_dataset.collate_fn)
        self.testing_loader = DataLoader(self.testing_dataset, batch_size=32,
                                      collate_fn=self.testing_dataset.collate_fn)

    def forward_batch(self, batch_data):
        token_ids = batch_data["input_ids"]
        target_labels = batch_data["label"]

        computed_loss, prediction_logits = self.neural_model(token_ids, target_labels)
        predicted_tags = prediction_logits.argmax(dim=-1)

        valid_mask = (target_labels != -100)
        true_tags = target_labels[valid_mask].tolist()
        pred_tags = predicted_tags[valid_mask].tolist()

        return computed_loss, true_tags, pred_tags