from huggingface_hub import login
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
from datasets import Dataset
import pickle
import evaluate
import time
from sklearn.metrics import f1_score
import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description="PyTorch Training Script with Arguments")
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train the model")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for DataLoader")
parser.add_argument("--lr", type=float, default=1e-5, help="learning rate for the model")
parser.add_argument("--dropout", type=float, default=0.3, help="dropout")
parser.add_argument("--c", type=float, default=0.5, help="percentile")
args = parser.parse_args()

print("HYPERPARAMETERS@@@@@@@@@ learning rate",args.lr, " batch size :",args.batch_size," number of epochs ", args.epochs, " percentile ", args.c )

# use your own token
my_HF_token = ''
login(token=my_HF_token)


# 2. Load Tokenizer and Model
model_name = "meta-llama/Llama-3.2-3B"
#adapters_path = "./checkpoint-190000" 
#adapters_path = "./plau_results_20250302-062236/checkpoint-35000"
adapters_path = "./plau_results_20250302-154807/checkpoint-25000"
#adapters_path = "./plau_results_20250225-011718/checkpoint-205000"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=11)
peft_config = PeftConfig.from_pretrained(adapters_path)
print(f"Loaded PEFT Config: {peft_config}")

# Integrate LoRA weights with the model
model = PeftModel.from_pretrained(model, adapters_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device ",device)
#device = "mps"
model = model.to(device)

# Set a padding token if not already defined
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Use the EOS token as a padding token
    model.config.pad_token_id = tokenizer.pad_token_id

def constructDataset(texts,labels):
    dataset = []
    for this_text, this_label in zip(texts,labels):
        dataset.append({"text":this_text, "labels":this_label})
    return dataset


with open('au_logits3.pkl', 'rb') as file:
       logits = pickle.load(file)

with open('../data/unlabelled_sentences.pkl','rb') as file:
       texts = pickle.load(file)

print("the texts length is ", len(texts))
print("logits length is ",len(logits))

thresholds = {}
probabilities = np.array(logits.cpu())
for i in range(probabilities.shape[1]):
    #thresholds[i] = {
    #'positive': np.percentile(probabilities[:, i], 100-args.c),
    #'negative': np.percentile(probabilities[:, i], args.c)
    #}
    thresholds[i] = {
    'positive': args.c,
    'negative': 1e-7
    }

print("positive threshold is ", args.c, " negative threshold is 1e-7")
#print("Thresholds for each label:", thresholds)

# Initialize lists for pseudo-labeled data and masks
pseudo_labels = []

# Iterate through each instance
for idx, probs in enumerate(logits):
    instance_labels = []

# Process each label in the instance
    for i, prob in enumerate(probs):
        if prob > thresholds[i]['positive']:
           if prob < 0.95 :
              instance_labels.append(1)
           else:
              instance_labels.append(-1)# Positive label
        elif prob <= thresholds[i]['negative']:
           instance_labels.append(0)  # Negative label
        else:
           instance_labels.append(-1)  # Low confidence, ignore
    pseudo_labels.append(instance_labels)
print("pseudo_labels ",len(pseudo_labels))

label_summary = {f"Label {i+1}": {"0s": 0, "1s": 0, "-1s": 0} for i in range(11)}

for labels in pseudo_labels:
    for i, label in enumerate(labels):
        if label == 0:
            label_summary[f"Label {i+1}"]["0s"] += 1
        elif label == 1:
            label_summary[f"Label {i+1}"]["1s"] += 1
        elif label == -1:
            label_summary[f"Label {i+1}"]["-1s"] += 1
# Output label summary
print("\nLabel Summary:")
for label, counts in label_summary.items():
    print(f"{label}: {counts}")

filtered_pseudo_labels = []
filtered_text = []
for labels, t in zip(pseudo_labels,texts) :
  if not all(label == -1 for label in labels):
    filtered_pseudo_labels.append(labels)
    filtered_text.append(t)

with open('../data/documents_train.pkl', 'rb') as file:
  texts = pickle.load(file)
with open('../data/labels_train.pkl', 'rb') as file:
  labels = pickle.load(file)

print("len of filtered data ", len(filtered_pseudo_labels))



train = constructDataset(texts+filtered_text,labels+filtered_pseudo_labels)

#with open('../data/documents_valid.pkl', 'rb') as file:
#  texts = pickle.load(file)
#with open('../data/labels_valid.pkl', 'rb') as file:
#  labels = pickle.load(file)
df = pd.read_csv('../data/val.csv')
labels = df.iloc[:,-11:].apply(lambda row: row.tolist(), axis=1).tolist()
texts = df.sentence.tolist()
val = constructDataset(texts,labels)

#with open('../data/documents_test.pkl', 'rb') as file:
#  texts = pickle.load(file)
#with open('../data/labels_test.pkl', 'rb') as file:
#  labels = pickle.load(file)
df = pd.read_csv('../data/test.csv')
labels = df.iloc[:,-11:].apply(lambda row: row.tolist(), axis=1).tolist()
texts = df.sentence.tolist()
test = constructDataset(texts,labels)

data = {"train": train, "validation":val, "test":test}

# 1. Load Data
# Replace this with your dataset loading and preprocessing logic.
def preprocess_function(examples):
    # Tokenize the inputs
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=60)


for item in data["train"]:
    for key in item.keys():
        if isinstance(item[key], float):
            print("error on key ", key)
            print("item[key] ",item[key])
            item[key] = str(item[key])  # Convert float to string


train_dataset = Dataset.from_list(data["train"])
validation_dataset = Dataset.from_list(data["validation"])
test_dataset = Dataset.from_list(data["test"])

train_dataset = train_dataset.map(lambda x: {**preprocess_function(x)})
validation_dataset = validation_dataset.map(lambda x: {**preprocess_function(x)})
test_dataset = test_dataset.map(lambda x: {**preprocess_function(x)})

# 4. Define Custom Loss Function
class MaskedBCELoss(nn.Module):
    def forward(self, logits, labels, label_mask):
        loss = nn.BCEWithLogitsLoss(reduction="none")(logits, labels.float())
        return (loss * label_mask).sum() / label_mask.sum()

loss_fn = MaskedBCELoss()


# 5. Trainer Setup
def compute_metrics(eval_pred):
    logits, all_labels = eval_pred
    all_predictions = (torch.sigmoid(torch.tensor(logits)) > 0.5).int()

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

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):

        labels = inputs.pop("labels")
        label_mask = labels != -1
        for key, val in inputs.items():
            if torch.is_tensor(val):
                inputs[key] = val.to(self.args.device)

        outputs = model(**inputs)
        logits = outputs.logits
        loss = loss_fn(logits, labels, label_mask)
        return (loss, outputs) if return_outputs else loss


    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        labels = inputs.pop("labels")
        label_mask = labels != -1
        for key, val in inputs.items():
            if torch.is_tensor(val):
                inputs[key] = val.to(self.args.device)

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        if logits.shape[0] != labels.shape[0]:
            raise ValueError(f"Logits batch size ({logits.shape[0]}) does not match labels batch size ({labels.shape[0]}).")

        return (None, logits, labels) if prediction_loss_only else (None, logits, labels)


#was 5000
print(f"./plau2_results_{time.strftime('%Y%m%d-%H%M%S')}")
training_args = TrainingArguments(
    output_dir=f"./plau_results_{time.strftime('%Y%m%d-%H%M%S')}",
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=5000,
    eval_steps=5000,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=args.epochs,
    logging_dir="./logs",
    logging_steps=5000,
    load_best_model_at_end=True,
    dataloader_drop_last=True,
    learning_rate=args.lr,
    metric_for_best_model="eval_macro_f1",
    greater_is_better=True
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model("./plau_fine_tuned_llama_lora_3b")

# Save the full model and tokenizer
model_dir = f"./llama3b_au_ftt_model_{time.strftime('%Y%m%d-%H%M%S')}"
model.save_pretrained(model_dir)  # Save the base model + adapter + classification head
tokenizer.save_pretrained(model_dir)  # Save the tokenizer configuration
print(f"Full model and tokenizer saved to: {model_dir}")

trainer.eval_dataset = test_dataset

# Evaluate the model
test_results = trainer.evaluate()

# Print the evaluation results
print("Test Evaluation Results:")
for key, value in test_results.items():
    print(f"{key}: {value}")
