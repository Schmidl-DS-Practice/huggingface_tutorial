from transformers import pipeline
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification) # with tensorflow: TFAutoModelForSequenceClassification
import torch
F = torch.nn.functional

# https://huggingface.co/docs/transformers/main_classes/pipelines
# https://huggingface.co/models

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# classifier = pipeline("sentiment-analysis")
# classifier = pipeline("sentiment-analysis", model=model_name)
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

res = classifier("""
                 In this video I show you everything to get started with Huggingface and the Transformers library.
                 We build a sentiment analysis pipeline, I show you the Model Hub, and how you can fine tune your
                 own models.
                 """)

text = ["The Huggingface transformers library is probably the most popular NLP library in Python right now.",
        "It can be combined directly with PyTorch or TensorFlow.",
        "It provides state-of-the-art Natural Language Processing models and has a very clean API that makes it extremely simple to implement powerful NLP pipelines."]
res = classifier(text)

# print(res)

tokens = tokenizer.tokenize("The Huggingface transformers library is probably the most popular NLP library in Python right now.")
token_ids = tokenizer.convert_tokens_to_ids(tokens)
input_ids = tokenizer("The Huggingface transformers library is probably the most popular NLP library in Python right now.")

# print(f"Token: {tokens}")
# print(f"token IDs: {token_ids}")
# print(f"input IDs: {input_ids}")


X_train = ["The Huggingface transformers library is probably the most popular NLP library in Python right now.",
           "It can be combined directly with PyTorch or TensorFlow."]
batch = tokenizer(X_train, padding=True,
                  truncation=True,
                  max_length=512,
                  return_tensors="pt")
# print(batch)

with torch.no_grad():
    # outputs = model(**batch) # tf > batch
    outputs = model(**batch, labels=torch.tensor([1,0]))
    print(outputs)
    predictions = F.softmax(outputs.logits, dim=1)
    print(predictions)
    labels = torch.argmax(predictions, dim=1)
    print(labels)
    labels = [model.config.id2label[label_id] for label_id in labels.tolist()]
    print(labels)

save_directory = "saved"
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)

tokenizer = AutoTokenizer.from_pretrained(save_directory)
model = AutoModelForSequenceClassification.from_pretrained(save_directory)




