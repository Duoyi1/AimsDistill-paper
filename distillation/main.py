import torch
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
import pickle
import random
import numpy as np
import torchmetrics
from torchmetrics.classification import Accuracy, F1Score
import argparse
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from dataset import *
from models2 import *
from sklearn.metrics import f1_score
import pandas as pd
import time
import gc
import ast

def distillation_loss(student_logits, teacher_logits, labels, mask, temperature, alpha, device):
    """
    Compute knowledge distillation loss for multi-label classification with missing labels.
    """

    eps = 1e-6  # Small value to prevent log(0) errors

    # **Corrected Teacher Probabilities (Ensures Sum=1)**
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    teacher_probs = teacher_probs.clamp(eps, 1.0)  # Avoid zero values

    # **Corrected Student Log Probabilities**
    student_probs = F.log_softmax(student_logits / temperature, dim=1)

    # **Compute KL Divergence (Masked)**
    kd_loss = F.kl_div(student_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)
    # **Mask out missing labels (-1)**
    kd_loss = kd_loss * mask.float()
    
    # **Prevent division by zero**
    if mask.sum() > 0:
        kd_loss = kd_loss.sum() / mask.sum()
    else:
        kd_loss = torch.tensor(0.0, device=student_logits.device)

    # **Debugging: If KL Loss is Negative, Print Logits & Probabilities**
    if kd_loss.item() < 0:
        with torch.no_grad():
            print(f"🚨 KL Loss is negative: {kd_loss.item()}")
            print("🔹 Teacher Probabilities (First 5):", teacher_probs[:5])
            print("🔹 Student Probabilities (First 5):", student_probs[:5])
            print("🔹 Logits - Teacher:", teacher_logits[:5])
            print("🔹 Logits - Student:", student_logits[:5])

    # **Binary Cross-Entropy Loss (Handles Multi-Label Classification)**
    bce_loss = nn.BCEWithLogitsLoss(reduction="none")
    labels_for_loss = labels.clone()
    labels_for_loss[~mask] = 0  # Replace -1 with 0 where mask is False

    ce_loss = bce_loss(student_logits, labels_for_loss)
    ce_loss = (ce_loss * mask.float()).sum() / mask.sum()

    # **Final Distillation Loss**
    final_loss = alpha * kd_loss + ce_loss

    #torch.cuda.empty_cache()

    return final_loss

def train_batch(batch, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_ids, attention_mask = [b.to(device) for b in batch]  # Move to GPU
    logits = model(input_ids=input_ids, attention_mask=attention_mask)
    return logits

def evaluate(model, data_loader, loss_fn,batch_size, length):
    model.eval() 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    total_loss = 0
    all_labels = []
    all_predictions = []
    with torch.no_grad():  
        for batch, labels,_ in tqdm(data_loader):
            logits = train_batch(batch, model)
            loss = loss_fn(logits.cpu(), labels.cpu()).mean()
            total_loss += loss.item() * batch_size
            
            #_, preds = torch.max(logits.cpu(), dim=1)
            predictions = (logits > 0.5).float() 
            all_labels.extend(labels.cpu().numpy())  # Collect true labels
            all_predictions.extend(predictions.cpu().numpy())  # Collect predicted labels

    # Calculate the final metrics
    avg_loss = total_loss / length

    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    
    micro_f1 = f1_score(all_labels, all_predictions, average='micro')
    weighted_f1 = f1_score(all_labels, all_predictions, average='weighted')
    macro_f1 = f1_score(all_labels, all_predictions, average='macro')

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
    f1_per_class = f1_score(all_labels, all_predictions, average=None)
    print("%%%%%%%%%%%% Per Class F1 Scores")
    for i, label_name in enumerate(list_name):
        print(f"F1-score for class {label_name}: {f1_per_class[i]:.4f}")    

# Print macro F1-score
    print(f"Macro F1-score: {macro_f1:.4f}")

    return avg_loss, macro_f1
    

def main(args):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  epochs = args.epochs
  lr = args.lr
  bce_loss = nn.BCEWithLogitsLoss(reduction="none")
  batch_size = args.batch_size
  max_length = 60

  random_seed = random.randint(1, 10000)
  #random_seed = 2702
  print("random_seed ",random_seed)
  torch.manual_seed(random_seed)
  random.seed(random_seed)
  np.random.seed(random_seed)
  
  #model_name = "answerdotai/ModernBERT-large"
  if args.dataset == "au":
     model_name = "../pre-train/modernbert-mlm"
  elif args.dataset == "uk":
     model_name = "../pre-train/modernbert-mlm-uk"
  elif args.dataset == "ca":
     #model_name = "answerdotai/ModernBERT-large"
     model_name = "../pre-train/modernbert-mlm-ca10"
     #model_name = "../rl/fine_tuned_class_20250307-115045.pth" 
  else:
      print("ERROR ON LOAD ING")
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  df = pd.read_csv("../data/au_train_context.csv")
  labels = df['targets'].tolist()
  labels = [ast.literal_eval(l) for l in labels]
  texts = df['sentence'].tolist()

  #teacher_logits = torch.load("../logits/modernbert.pt")
  
  def load_tensor(path):
    obj = torch.load(path, map_location=torch.device("cpu"),weights_only=False)
    if isinstance(obj, np.ndarray):
        return torch.tensor(obj, dtype=torch.float32)
    elif isinstance(obj, torch.Tensor):
        return obj.to(torch.float32)
    else:
        raise TypeError(f"Unsupported type: {type(obj)} in {path}")
  
  #weights = np.random.dirichlet(np.ones(6), size=1)[0]
  #weights = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
  weights = [0.45217772, 0.15573896, 0.27859665, 0.05981953, 0.02509557, 0.02857157]
  print("Dirichlet Weights:", weights)
  #print("Sum:", weights.sum())


  teacher_logits = (
    weights[0] * load_tensor("../logits/prompt_logits.pt") +
    weights[1] * load_tensor("../logits/train_" + args.dataset + "_logits.pt") +
    weights[2] * load_tensor("../logits/cl2_5_6.pt") +
    weights[3] * load_tensor("../logits/cl2_7_8.pt") +
    weights[4] * load_tensor("../logits/cl2_7_9.pt") +
    weights[5] * load_tensor("../logits/context_logits.pt")
  )

  #teacher_logits = teacher_logits/5

  print("teacher_logits ",teacher_logits.size())

  train_dataset = MyDataset(texts, tokenizer, max_length, labels, teacher_logits = teacher_logits)
  print("length of the training dataset is ", len(train_dataset))

  df = pd.read_csv('../data/val.csv')
  labels = df.iloc[:,-11:].apply(lambda row: row.tolist(), axis=1).tolist()
  texts = df.sentence.tolist()
  val_dataset = MyDataset(texts, tokenizer, max_length, labels)

  best = 0.72
  if args.dataset == "au":
     df = pd.read_csv('../data/test.csv')
     labels = df.iloc[:,-11:].apply(lambda row: row.tolist(), axis=1).tolist()
     texts = df.sentence.tolist()
     test_dataset = MyDataset(texts, tokenizer, max_length, labels)
     factor = 1
  elif args.dataset == "ca":
     df = pd.read_csv("../data/Canadian.csv", encoding='latin1').iloc[:3658,:]
     texts = df.sentence.tolist()
     labels = df.targets
     labels = [ast.literal_eval(i) for i in labels]
     test_dataset = MyDataset(texts, tokenizer, max_length, labels)
  elif args.dataset == "uk":
     df = pd.read_csv("../data/UK.csv", encoding='latin1')
     texts = df.sentence.tolist()
     labels = df.targets
     labels = [ast.literal_eval(i) for i in labels]
     test_dataset = MyDataset(texts, tokenizer, max_length, labels)
     factor = 1
  else:
      print("@@@@@@@@!!!!!!!!!!WRONG DATASET NAME ###############")


  #model_name = 'distilbert-base-uncased' 

  # Randomly split the dataset (with reproducibility)
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

  # If you are training, here’s a basic training loop structure
  model = SentenceBERTForDownstreamTask(tokenizer, model_name,dropout=args.dropout).to(device)
  #model.load_state_dict(torch.load("../rl/fine_tuned_class_20250307-115045.pth"))
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  model.train()  # Set the model to train mode
  train_loss = []
  val_loss = []
  test_loss = []
  step = 0

  for epoch in range(epochs):
      epoch_loss = 0
      print("Next epoch...............")
      for batch, labels, teacher_logits in tqdm(train_loader):
          step+=1
          if step % 1000 == 0:
              _, f1 = evaluate(model, val_loader, bce_loss,batch_size,len(test_dataset))
              if f1*factor >best:
                 lr = lr*0.5
                 print("Reduce leanring rate to ",lr)
                 for param_group in optimizer.param_groups:
                     param_group['lr'] = lr
          labels = labels.to(device)
          #teacher_logits = teacher_logits.to(device)
          teacher_logits = teacher_logits.to(device, non_blocking=True)
          student_logits = train_batch(batch, model)
          mask = (labels != -1).to(device)
          loss = distillation_loss(student_logits, teacher_logits, labels, mask, args.temp, args.alpha, device)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          epoch_loss += loss.item()*labels.size()[0]
      
          del student_logits, teacher_logits, loss, batch, labels, mask
          if step % 10 == 0:  # Only run garbage collection every 50 batches
             torch.cuda.empty_cache()
             gc.collect()
     
      train_loss.append(epoch_loss/len(train_dataset))

      print("########################VAL")
      this_val_loss, f1 = evaluate(model, val_loader, bce_loss,batch_size, len(val_dataset))
      val_loss.append(this_val_loss)

      print("#######################3TEST")
      this_test_loss,f1 = evaluate(model, test_loader, bce_loss,batch_size,len(test_dataset))
      test_loss.append(this_test_loss)
      print(f"Epoch {epoch+1}/{epochs}, Test Loss: {this_test_loss}")
      print("Test mF1: ", f1)
 

  return f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Training Script with Arguments")

    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train the model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for DataLoader")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate for the model")
    parser.add_argument("--dropout", type=float, default=0.3, help="dropout")
    parser.add_argument("--temp", type=float, default=4.0, help="temprature")
    parser.add_argument("--alpha", type=float, default=0.1, help="alpha")
    parser.add_argument("--dataset", type=str, default="au", help="dataset")
    args = parser.parse_args()
    print("HYPERPARAMETERS@@@@@@@@@ learning rate",args.lr, " batch size :",args.batch_size," number of epochs ", args.epochs, " temprature ",args.temp, " alpha ", args.alpha)
    print("######## YOU ARE WORKING ON DATASET ", args.dataset)
    
    mf1 = main(args)
