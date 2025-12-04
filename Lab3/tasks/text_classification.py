import torch
from torch.utils.data import DataLoader

from data.uit_vsfc import UIT_VSFC
from model import lstm
from tasks.base_task import BaseTask


class TextClassificationTask(BaseTask):

    def __init__(self, vocabulary, training_path, validation_path, testing_path,
                 neural_model, model_checkpoint_path, learning_rate=1e-3):

        self.training_path = training_path
        self.validation_path = validation_path
        self.testing_path = testing_path
        self.neural_model = neural_model
        self.target_field = vocabulary.target_field
        self.model_checkpoint_path = model_checkpoint_path
        self.learning_rate = learning_rate

        super().__init__(vocabulary, neural_model, model_checkpoint_path)

    def load_datasets(self):
        self.training_dataset = UIT_VSFC(self.training_path, self.target_field, self.vocabulary)
        self.validation_dataset = UIT_VSFC(self.validation_path, self.target_field, self.vocabulary)
        self.testing_dataset = UIT_VSFC(self.testing_path, self.target_field, self.vocabulary)

    def create_dataloaders(self):
        self.training_loader = DataLoader(self.training_dataset, batch_size=32, shuffle=True,
                                       collate_fn=self.training_dataset.collate_fn)
        self.validation_loader = DataLoader(self.validation_dataset, batch_size=32,
                                     collate_fn=self.validation_dataset.collate_fn)
        self.testing_loader = DataLoader(self.testing_dataset, batch_size=32,
                                      collate_fn=self.testing_dataset.collate_fn)


    def forward_batch(self, batch_data):
        token_ids = batch_data["input_ids"]
        target_labels = batch_data["label"].view(-1)

        computed_loss, prediction_logits = self.neural_model(token_ids, target_labels)   

        predicted_classes = prediction_logits.argmax(dim=-1).tolist()  
        true_classes = target_labels.tolist()                  

        return computed_loss, true_classes, predicted_classes
