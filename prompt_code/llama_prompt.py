from huggingface_hub import login
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import pickle
import evaluate
import time
from sklearn.metrics import f1_score
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description="PyTorch Training Script with Arguments")
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train the model")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for DataLoader")
parser.add_argument("--lr", type=float, default=1e-5, help="learning rate for the model")
parser.add_argument("--dropout", type=float, default=0.3, help="dropout")
args = parser.parse_args()
print("HYPERPARAMETERS@@@@@@@@@ learning rate",args.lr, " batch size :",args.batch_size," number of epochs ", args.epochs )

# use your own token
my_HF_token = ''
login(token=my_HF_token)

# 2. Load Tokenizer and Model
model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=11)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device ",device)
#device = "mps"
model = model.to(device)

# Set a padding token if not already defined
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Use the EOS token as a padding token
    model.config.pad_token_id = tokenizer.pad_token_id

#prompt = "Classify the following sentence based on whether it includes specific, concrete information related to any of the reporting requirements listed below: Approval by the board. Signature by a responsible member of the organization. Description of the reporting entity’s structure, operations, or supply chains. Description of the risks of modern slavery practices in the operations and supply chains of the reporting entity and any entities it owns or controls. Description of the actions taken by the reporting entity and any entities it owns or controls to assess and address these risks, including due diligence and remediation processes. Description of the how the entity assesses the effectiveness of those actions. Sentence: [Insert sentence here]"
#prompt N
prompt = "Classify the following sentence into the appropriate categories. Approval by the board. Signature by a responsible member of the organization. Description of the reporting entity’s structure, operations, or supply chains. Description of the risks of modern slavery practices in the operations and supply chains of the reporting entity and any entities it owns or controls. Description of the actions taken by the reporting entity and any entities it owns or controls to assess and address these risks, including due diligence and remediation processes. Description of the how the entity assesses the effectiveness of those actions. Sentence:"

def constructDataset(texts,labels):
    dataset = []
    for this_text, this_label in zip(texts,labels):
        dataset.append({"text":prompt + this_text, "labels":this_label})
    return dataset

with open('../data/documents_train.pkl', 'rb') as file:
  texts = pickle.load(file)
with open('../data/labels_train.pkl', 'rb') as file:
  labels = pickle.load(file)
train = constructDataset(texts,labels)

df = pd.read_csv('../data/val.csv')
labels = df.iloc[:,-11:].apply(lambda row: row.tolist(), axis=1).tolist()
texts = df.sentence.tolist()
val = constructDataset(texts,labels)

df = pd.read_csv('../data/test.csv')
labels = df.iloc[:,-11:].apply(lambda row: row.tolist(), axis=1).tolist()
texts = df.sentence.tolist()
test = constructDataset(texts,labels)

data = {"train": train, "validation":val, "test":test}

# 1. Load Data
# Replace this with your dataset loading and preprocessing logic.
def preprocess_function(examples):
    # Tokenize the inputs
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=230)

train_dataset = Dataset.from_list(data["train"])
validation_dataset = Dataset.from_list(data["validation"])
test_dataset = Dataset.from_list(data["test"])

train_dataset = train_dataset.map(lambda x: {**preprocess_function(x)})
validation_dataset = validation_dataset.map(lambda x: {**preprocess_function(x)})
test_dataset = test_dataset.map(lambda x: {**preprocess_function(x)})



# 3. Integrate LoRA
# was 16
print("Lora 32")
lora_config = LoraConfig(task_type="SEQ_CLS",r=32, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.1, modules_to_save=["score"], )
model = get_peft_model(model, lora_config)

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
training_args = TrainingArguments(
    output_dir=f"./prompt_{time.strftime('%Y%m%d-%H%M%S')}",
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=2000,
    eval_steps=2000,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=args.epochs,
    logging_dir="./logs",
    logging_steps=2000,
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
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model("./prompt")

# Save the full model and tokenizer
model_dir = f"./llama1b_prompt_model_{time.strftime('%Y%m%d-%H%M%S')}"
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
