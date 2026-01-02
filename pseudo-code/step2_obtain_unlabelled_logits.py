from huggingface_hub import login
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model,PeftConfig,PeftModel
from datasets import Dataset
import pickle
import evaluate
from sklearn.metrics import f1_score
import time
import os
from safetensors.torch import load_file
import torch
import ast
import pandas as pd


# use your own token
my_HF_token = ''
login(token=my_HF_token)

# 2. Load Tokenizer and Model

model_name = "meta-llama/Llama-3.2-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, device_map="auto",torch_dtype="auto",offload_state_dict=False, num_labels=11)
#adapters_path = "./uk-checkpoint-15000"
#adapters_path = "./ca-checkpoint-95000"
#adapters_path = "./checkpoint-190000"
adapters_path = "./checkpoint-85000"


# Set the device
device = "cuda"
model = model.to(device)
print(next(model.parameters()).device)
max_length = 60

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Use the EOS token as a padding token
    model.config.pad_token_id = tokenizer.pad_token_id

# Load the LoRA configuration
peft_config = PeftConfig.from_pretrained(adapters_path)
print(f"Loaded PEFT Config: {peft_config}")

# Integrate LoRA weights with the model
model = PeftModel.from_pretrained(model, adapters_path)

# Ensure the model is on the correct device
model = model.to(device)

# Verify the loaded model
print(model)


def constructDataset(texts,labels):
    dataset = []
    for this_text, this_label in zip(texts,labels):
        dataset.append({"text":this_text, "labels":this_label})
    return dataset

#df = pd.read_csv('../data/val.csv')
#labels = df.iloc[:,-11:].apply(lambda row: row.tolist(), axis=1).tolist()
#texts = df.sentence.tolist()
#val = constructDataset(texts,labels)

#df = pd.read_csv('../data/test.csv')
#labels = df.iloc[:,-11:].apply(lambda row: row.tolist(), axis=1).tolist()
#texts = df.sentence.tolist()
#test = constructDataset(texts,labels)

#df = pd.read_csv("../data/Canadian.csv", encoding='latin1').iloc[:3658,:]
##print(df)
#texts = df.sentence.tolist()
#labels = df.targets
#labels = [ast.literal_eval(i) for i in labels]
#ca = constructDataset(texts,labels)
#print("len of texts ",len(texts))
#print("len of labels ",len(labels))

#with open('../data/uk_sentence.pkl', 'rb') as file:
#      texts = pickle.load(file)
#with open('../data/uk_labels.pkl', 'rb') as file:
#      labels = pickle.load(file)
#df = pd.read_csv("../data/UK.csv", encoding='latin1')
##print(df)
#texts = df.sentence.tolist()
#labels = df.targets
#labels = [ast.literal_eval(i) for i in labels]
#uk = constructDataset(texts,labels)

with open('../data/documents_train.pkl', 'rb') as file:
  texts = pickle.load(file)
with open('../data/labels_train.pkl', 'rb') as file:
  labels = pickle.load(file)
train = constructDataset(texts,labels)

#with open('../data/documents_test.pkl', 'rb') as file:
#  texts = pickle.load(file)
#with open('../data/labels_test.pkl', 'rb') as file:
#  labels = pickle.load(file)
#test = constructDataset(texts,labels)

#data = {"validation": val, "test":test, "ca":ca, "uk":uk}
data = {"train":train}

# 1. Load Data
# Replace this with your dataset loading and preprocessing logic.
def preprocess_function(examples):
    # Tokenize the inputs
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=max_length)

train_dataset = Dataset.from_list(data["train"])
#validation_dataset = Dataset.from_list(data["validation"])
#val_au_dataset = Dataset.from_list(data["au"])
#val_ca_dataset = Dataset.from_list(data["valca"])
#test_dataset = Dataset.from_list(data["test"])
#ca_dataset = Dataset.from_list(data["ca"])
#uk_dataset = Dataset.from_list(data["uk"])

train_dataset = train_dataset.map(lambda x: {**preprocess_function(x)})
#validation_dataset = validation_dataset.map(lambda x: {**preprocess_function(x)})
#val_au_dataset = val_au_dataset.map(lambda x: {**preprocess_function(x)})
#val_ca_dataset = val_ca_dataset.map(lambda x: {**preprocess_function(x)})
#test_dataset = test_dataset.map(lambda x: {**preprocess_function(x)}, batched=True)
#ca_dataset = ca_dataset.map(lambda x: {**preprocess_function(x)}, batched=True)
#uk_dataset = uk_dataset.map(lambda x: {**preprocess_function(x)}, batched=True)

from torch.utils.data import DataLoader
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
#validation_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
#val_au_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
#val_ca_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
#test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
#ca_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
#uk_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Create a DataLoader

train_dataloader = DataLoader(
    train_dataset, batch_size=128, shuffle=False  # Adjust batch_size as needed
)

#val_dataloader = DataLoader(
#    validation_dataset, batch_size=8, shuffle=False  # Adjust batch_size as needed
#)

#val_ca_dataloader = DataLoader(
#    val_ca_dataset, batch_size=8, shuffle=False  # Adjust batch_size as needed
#)

#val_au_dataloader = DataLoader(
#    val_au_dataset, batch_size=8, shuffle=False  # Adjust batch_size as needed
#)

#test_dataloader = DataLoader(
#    test_dataset, batch_size=8, shuffle=False  # Adjust batch_size as needed
#)

#uk_dataloader = DataLoader(
#    uk_dataset, batch_size=8, shuffle=False  # Adjust batch_size as needed
#)
#ca_dataloader = DataLoader(
#    ca_dataset, batch_size=8, shuffle=False  # Adjust batch_size as needed
#)

def compute_metrics(eval_pred):
    logits, all_labels = eval_pred
    all_predictions = (torch.sigmoid(torch.tensor(logits)) > 0.5).int()

    from sklearn.metrics import precision_score, recall_score
    macro_precision = precision_score(all_labels, all_predictions, average='macro')
    macro_recall = recall_score(all_labels, all_predictions, average='macro')
    print("macro_precision ",macro_precision ," macro_recall ",macro_recall)

    # Compute Macro F1
    macro_f1 = f1_score(all_labels, all_predictions, average='macro')

    # Compute F1 per class
    f1_per_class = f1_score(all_labels, all_predictions, average=None)
    list_name = [
        "approval",
        "signature",
        "c1 (reporting entity)",
        "c2 (structure)",
        "c2 (operations)",
        "c2 (supply chains)",
        "c3 (risk description)",
        "c4 (risk mitigation)",
        "c4 (remediation)",
        "c5 (effectiveness)",
        "c6 (consultation)"
    ]

    print("%%%%%%%%%%%% Per Class F1 Scores")
    for i, label_name in enumerate(list_name):
        print(f"F1-score for class {label_name}: {f1_per_class[i]:.4f}")
    print("macro f1 ", macro_f1)
    
    return {
        "macro_f1": macro_f1,
        "f1_per_class": f1_per_class.tolist(),  # Convert to list for JSON serialization
    }

import tqdm

all_logits = []
all_labels = []
print("############### TRAIN ########")
with torch.no_grad():
    for batch in tqdm.tqdm(train_dataloader):
        inputs = {key: val.to(device) for key, val in batch.items() if key != "labels"}
        labels = batch["labels"].to(device)
        outputs = model(**inputs)
        logits = outputs.logits
        all_logits.append(logits)
all_logits = torch.cat(all_logits)

torch.save(all_logits, "../logits/train_au_logits.pt", pickle_protocol=4)
print("DONE@@@@@@@@@@@@@@$$$$$$$$$$$$$$$$$$")
#all_logits = []
#all_labels = []
#print("############### CA")
#with torch.no_grad():
#    for batch in tqdm.tqdm(ca_dataloader):
#        inputs = {key: val.to(device) for key, val in batch.items() if key != "labels"}
#        labels = batch["labels"].to(device)
#        outputs = model(**inputs)
#        logits = outputs.logits
#        all_logits.append(logits)
#        all_labels.append(labels)
#all_logits = torch.cat(all_logits)
#all_labels = torch.cat(all_labels)
#compute_metrics((all_logits.cpu(),all_labels.cpu()))


#all_logits = []
#all_labels = []
#print("####################### VA")
#with torch.no_grad():
#    for batch in tqdm.tqdm(val_dataloader):
#        inputs = {key: val.to(device) for key, val in batch.items() if key != "labels"}
#        labels = batch["labels"].to(device)
#        outputs = model(**inputs)
#        logits = outputs.logits
#        all_logits.append(logits)
#        all_labels.append(labels)
#all_logits = torch.cat(all_logits)
#all_labels = torch.cat(all_labels)
#compute_metrics((all_logits.cpu(),all_labels.cpu()))

#all_logits = []
#all_labels = []
#print("####################### CA")
#with torch.no_grad():
#    for batch in tqdm.tqdm(ca_dataloader):
#        inputs = {key: val.to(device) for key, val in batch.items() if key != "labels"}
#        labels = batch["labels"].to(device)
#        outputs = model(**inputs)
#        logits = outputs.logits
#        all_logits.append(logits)
#        all_labels.append(labels)
#all_logits = torch.cat(all_logits)
#all_labels = torch.cat(all_labels)
#compute_metrics((all_logits.cpu(),all_labels.cpu()))


#all_logits = []
#all_labels = []
#print("################ UK")
## Loop through the DataLoader
#with torch.no_grad():
#    for batch in tqdm.tqdm(uk_dataloader):
#        # Move inputs and labels to the device
#        inputs = {key: val.to(device) for key, val in batch.items() if key != "labels"}
#        labels = batch["labels"].to(device)

#        # Get model outputs
#        outputs = model(**inputs)
#        logits = outputs.logits
       

#        # Store logits and labels for metric calculation
#        all_logits.append(logits)
#        all_labels.append(labels)

# Concatenate all logits and labels
#all_logits = torch.cat(all_logits)
#all_labels = torch.cat(all_labels)
#compute_metrics((all_logits.cpu(),all_labels.cpu()))




#all_logits = []
#all_labels = []
#print("Test set of AU test set")
## Loop through the DataLoader
#with torch.no_grad():
#    for batch in tqdm.tqdm(test_dataloader):
#        # Move inputs and labels to the device
#        inputs = {key: val.to(device) for key, val in batch.items() if key != "labels"}
#        labels = batch["labels"].to(device)

#        # Get model outputs
#        outputs = model(**inputs)
#        logits = outputs.logits

#        # Store logits and labels for metric calculation
#        all_logits.append(logits)
#        all_labels.append(labels)

## Concatenate all logits and labels
#all_logits = torch.cat(all_logits)
#all_labels = torch.cat(all_labels)
#compute_metrics((all_logits.cpu(),all_labels.cpu()))
