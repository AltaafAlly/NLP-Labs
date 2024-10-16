# %%
from datasets import load_dataset

# Load the Swahili news dataset
dataset = load_dataset('community-datasets/swahili_news')

# %%
import re

def clean_text(example):
    text = example['text']
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove non-alphabetic characters except spaces
    text = re.sub(r'[^a-zA-ZäöüÄÖÜßẞ\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Convert to lowercase
    text = text.lower()
    example['text'] = text
    return example

# Apply the cleaning function
dataset = dataset.map(clean_text)


# %%
# Get the unique labels
unique_labels = dataset['train'].unique('label')
print("Unique Labels:", unique_labels)

# %%
def map_labels_to_binary(example):
    if example['label'] == 1:
        example['label'] = 1
    else:
        example['label'] = 0
    return example

# Apply the mapping function
dataset = dataset.map(map_labels_to_binary)


# %%
from collections import Counter

# Calculate label distribution
label_counts = Counter(dataset['train']['label'])
print("Label Distribution in Training Set:", label_counts)


# %%
from datasets import DatasetDict

# Assuming the dataset has only a 'train' split, we'll create our own splits
dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)
test_valid = dataset['test'].train_test_split(test_size=0.5, seed=42)

# Create a DatasetDict
dataset = DatasetDict({
    'train': dataset['train'],
    'validation': test_valid['train'],
    'test': test_valid['test'],
})


# %%
from transformers import AutoTokenizer

# Load the tokenizer from the first fine-tuning step
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

def tokenize_function(batch):
    return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=128)

# Tokenize the datasets
tokenized_datasets = dataset.map(tokenize_function, batched=True)


# %%
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])


# %%
from transformers import AutoModelForSequenceClassification

# Load the model from the first fine-tuning step
model = AutoModelForSequenceClassification.from_pretrained(
    'xlm-roberta-base',
    num_labels=2,
)


# %%
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir='/datasets/mdawood/results_binary_classification-base',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=8,  # Adjust based on your GPU memory
    per_device_eval_batch_size=8,
    num_train_epochs=3,             # Adjust based on your needs
    weight_decay=0.01,
    logging_dir='/datasets/mdawood/logs_binary_classification-base',
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model='f1',
)


# %%
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Convert logits to probabilities using softmax
    probs = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    # Get the predicted class (0 or 1)
    predictions = np.argmax(probs, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }


# %%
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    compute_metrics=compute_metrics,
)


# %%
trainer.train()

# %%
# Save the fine-tuned model
trainer.save_model('/datasets/mdawood/swahili-xlmr-binary-classification-base')

# Save the tokenizer
tokenizer.save_pretrained('/datasets/mdawood/swahili-xlmr-binary-classification-base')

# %%
test_results = trainer.evaluate(eval_dataset=tokenized_datasets['test'])
print("Test Results:", test_results)


# %%
print(f"Test Accuracy: {test_results['eval_accuracy']:.4f}")
print(f"Test F1 Score: {test_results['eval_f1']:.4f}")
print(f"Test Precision: {test_results['eval_precision']:.4f}")
print(f"Test Recall: {test_results['eval_recall']:.4f}")


# %%
import torch

# Get a batch of test examples
test_samples = tokenized_datasets['test'][:5]  # Adjust as needed

# Move inputs to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
inputs = {k: v.to(device) for k, v in test_samples.items() if k in ['input_ids', 'attention_mask']}

# Get outputs with attentions
with torch.no_grad():
    outputs = model(**inputs, output_attentions=True)
    attentions = outputs.attentions  # Tuple of attention tensors

# Process attentions for visualization



