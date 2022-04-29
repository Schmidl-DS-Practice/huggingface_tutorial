from transformers import pipeline
import torch
F = torch.nn.functional

# https://huggingface.co/docs/transformers/main_classes/pipelines
classifier = pipeline("sentiment-analysis")
res = classifier("""
                 In this video I show you everything to get started with Huggingface and the Transformers library.
                 We build a sentiment analysis pipeline, I show you the Model Hub, and how you can fine tune your
                 own models.
                 """)



