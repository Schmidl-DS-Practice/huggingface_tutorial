from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
Dataset = torch.utils.data.Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, TrainingArguments
from transformers import Trainer, TraningArguments

# Prepare dataset
# Load pretrained Tokenizer, call it with a datset > encoding
# Build PyTorch Dataset with encodings
# Load pretrained Model
# Load Trainer and train it or use native PyTorch training pipeline

model_name = "distilbert-base-uncased"

def read_imdb_split(split_dir):
    split_dir = Path(split_dir)
    texts = []
    labels = []
    for label_dir in ["pos", "neg"]:
        for text_file in (split_dir/label_dir).iterdir():
            texts.append(text_file.read_text())
            labels.append(0 if label_dir == "neg" else 1)

    return texts, labels

# Large Movie Review Dataset
# http://ai.stanford.edu/~amaas/data/sentiment

train_texts, train_labels = read_imdb_split("aclimdb/train")
test_texts, tests_label = read_imdb_split("aclimdb/train")

train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)

class IMDbDataset(Dataset):
    def __init__(self, encodings, labels) -> None:
        super().__init__()
        self.encodings = encodings
        self.labels = labels

    def __getitem(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in  self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

# ensure that all of our sequences are padded to the same length and are truncated to be no longer than model's
# maximum input length. This will allow us to feed batches of sequences into the model at the same time.
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

train_dataset = IMDbDataset(train_encodings, train_labels)
val_dataset = IMDbDataset(val_encodings, val_labels)
test_dataset = IMDbDataset(test_encodings, tests_label)

training_args = TrainingArguments(output_dir="./results",
                                  num_train_epochs=2,
                                  per_device_train_batch_size=16,
                                  per_device_eval_batch_size=64,
                                  warmup_steps=500,
                                  learning_rate=5e-5,
                                  weight_decay=.01,
                                  logging_dir="./logs",
                                  logging_steps=10)

model = DistilBertForSequenceClassification.from_pretrained(model_name)

trainer = Trainer(model=model,
                  args=training_args,
                  train_dataset=train_dataset,
                  eval_dataset=val_dataset)

trainer.train()

## or native PyTorch
Dataset = torch.utils.data
