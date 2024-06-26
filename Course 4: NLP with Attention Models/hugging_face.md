# Hugging Face: 

- Hugging face is a data science platform where we can download pretrained models for various ML tasks.
- In particular we can use the platform to access various resources like datasets, transformers, BERT, tokenizers, preprocessors for text etc. 
- Doing so we can test various models and fine tune them to our use case. 
- You can also access various models trained on different datasets that might be more suitable to our use case. 
- Hugging face works well with PyTorch/Tensorflow.


## Components on Hugging Face:
- HF provide many useful APIs that we can use. 

### Pipelines:
- Pipelines are the equivalent of using a pretrained model for inference, which applies:
    - The preprocessing to inputs. (has a tokenizer in the background to clean/process text)
    - Pass through the model.
    - Post processing of the outputs. 
- Eg: if we wanted to create a question/answering system. 

### Model Checkpoints:
- The pretrained models have a checkpoint which references at what point the model training stopped. The prevents overfitting. We can think of this as a snapshot of a model. 
- We can download models based on the dataset, eg: DistilBERT references a BERT model trained on question/answering (SQuAD: Stanford Question Answering Dataset. )