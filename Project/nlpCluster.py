# %%
#Imports
import requests
import lzma
import os
from datasets import Dataset, DatasetDict
import re
from transformers import AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, DataCollatorForLanguageModeling, Trainer
import numpy as np

# %%
# url = "http://data.statmt.org/cc-100/sw.txt.xz"
# file_name = "sw.txt.xz"
# response = requests.get(url, stream=True)
# with open(file_name, "wb") as file:
#     for chunk in response.iter_content(chunk_size=1024):
#         if chunk:
#             file.write(chunk)
# print(f"Downloaded {file_name}")

# output_file = "sw.txt"
# with lzma.open(file_name, "rb") as compressed_file:
#     with open(output_file, "wb") as extracted_file:
#         extracted_file.write(compressed_file.read())
# print(f"Extracted to {output_file}")
# os.remove(file_name)

# %%
# Step 2: Read and prepare data
num_lines_to_read = 2500000  # Adjust this as needed
text_data = []
with open('/datasets/mdawood/sw.txt', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i < num_lines_to_read:
            line = line.strip()
            if line:
                text_data.append(line)
        else:
            break

# %%
# Create dataset
data_dict = {'text': text_data}
dataset = Dataset.from_dict(data_dict)


# %%
# Clean text
def clean_text(example):
    text = example['text']
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-ZäöüÄÖÜßẞ\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    return {'text': text}



# %%
dataset = dataset.map(clean_text)
dataset = dataset.shuffle(seed=42)


# %%
# Split dataset (80% train, 10% validation, 10% test)
split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
test_valid = split_dataset['test'].train_test_split(test_size=0.5, seed=42)
swahili_dataset = DatasetDict({
    'train': split_dataset['train'],
    'validation': test_valid['train'],
    'test': test_valid['test'],
})


# %%
# Print the number of samples in each split
print(f"Number of samples in train: {len(swahili_dataset['train'])}")
print(f"Number of samples in validation: {len(swahili_dataset['validation'])}")
print(f"Number of samples in test: {len(swahili_dataset['test'])}")

# %%
# Tokenization
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

def tokenize_function(batch):
    return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=128)

tokenized_datasets = swahili_dataset.map(tokenize_function, batched=True, num_proc=4)
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask'])



# %%

# Pre-trained model loading
model = AutoModelForMaskedLM.from_pretrained('xlm-roberta-base')


# %%
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)


# %%
# Step 3: Define the Trainer for pre-fine-tuning perplexity
training_args_before = TrainingArguments(
    output_dir='/datasets/mdawood/results',
    per_device_eval_batch_size=8,
    logging_dir='/datasets/mdawood/logs',
    logging_steps=500,
    evaluation_strategy="no"  # No training, just evaluation
)

# %%
# Create a Trainer instance for evaluation before fine-tuning
trainer_before = Trainer(
    model=model,
    args=training_args_before,
    eval_dataset=tokenized_datasets['validation'],  # Use validation set for evaluation
    data_collator=data_collator,
)

# %%
# Step 4: Evaluate pre-trained model (before fine-tuning)
eval_results_before = trainer_before.evaluate()
perplexity_before = np.exp(eval_results_before['eval_loss'])
print(f"Perplexity before fine-tuning: {perplexity_before:.2f}")

# %%
# Step 5: Fine-tune the model on Swahili dataset
training_args = TrainingArguments(
    output_dir='/datasets/mdawood/results',
    overwrite_output_dir=True,
    evaluation_strategy='epoch',
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir='/datasets/mdawood/logs',
    logging_steps=500,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    data_collator=data_collator,
)

# %%
trainer.train()


# %%
# Step 6: Calculate perplexity after fine-tuning
eval_results_after = trainer.evaluate()
perplexity_after = np.exp(eval_results_after['eval_loss'])
print(f"Perplexity after fine-tuning: {perplexity_after:.2f}")


# %%
# Step 7: Save the fine-tuned model
trainer.save_model('/datasets/mdawood/swahili-xlmr-finetuned')
tokenizer.save_pretrained('/datasets/mdawood/swahili-xlmr-finetuned')


