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
import torch.nn.functional as F
import time
import ast

def contrastive_loss(embeddings, labels, labelA, labelB, margin=0.5):
    """
    Contrastive loss focusing only on separating Label 0 and Label 1.

    Args:
        embeddings (torch.Tensor): Shape (batch_size, hidden_dim).
        labels (torch.Tensor): Binary multi-label tensor of shape (batch_size, num_labels).
        margin (float): Margin for contrastive loss.

    Returns:
        torch.Tensor: Loss value.
    """
    batch_size = embeddings.shape[0]
    loss = 0.0
    num_pairs = 0

    # Compute pairwise cosine similarity
    embeddings = F.normalize(embeddings, p=2, dim=1)
    similarity_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)

    # Iterate over all pairs
    for i in range(batch_size):
        for j in range(batch_size):
            if i == j:
                continue
            label_0_i = int(labels[i, labelA].to(torch.int).item())  # Ensure integer comparison
            label_0_j = int(labels[j, labelA].to(torch.int).item())
            label_1_i = int(labels[i, labelB].to(torch.int).item())  # Ensure integer comparison
            label_1_j = int(labels[j, labelB].to(torch.int).item())

            # If both have Label 0 (positives only), pull them together
            if label_0_i == 1 and label_0_j == 1:
                loss += 1 - similarity_matrix[i, j]  # Pull closer

            # If both have Label 1 (positives only), pull them together
            elif label_1_i == 1 and label_1_j == 1:
                loss += 1 - similarity_matrix[i, j]  # Pull closer
            elif (label_1_i == 1 and label_1_j == 0) or (label_1_i == 0 and label_1_j == 1):
                loss += F.relu(margin - similarity_matrix[i, j])  # Push apart
            num_pairs += 1

    return loss / max(num_pairs, 1)  # Normalize by number of pairs


def train_batch(batch, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_ids, attention_mask = [b.to(device) for b in batch]  # Move to GPU
    logits = model(input_ids=input_ids, attention_mask=attention_mask)
    return logits

def evaluate(model, data_loader, loss_fn,batch_size, length, da):
    model.eval() 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #accuracy_metric = Accuracy(task="multiclass", num_classes=12)  # Change num_classes based on your task
    #mf1_metric = F1Score(task="multiclass", num_classes=12, average='macro')
    #wf1_metric = F1Score(task="multiclass", num_classes=12, average='weighted')
    total_loss = 0
    all_labels = []
    all_predictions = []
    with torch.no_grad():  
        for batch, labels in tqdm(data_loader):
            logits,_ = train_batch(batch, model)
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
    print("%%%%%%%%%%%% per class")
    for i, label in enumerate(range(11)):
        print(f"F1-score for class {list_name[i]}: {f1_per_class[i]:.4f}")

    print("#### Target class ##########")
    good_result = 0
    delta = 0.00
    for i, label in enumerate(range(11)):
        if i in args.label:
          if da == "AU":
            if i == 5 and f1_per_class[i]>(0.7537-delta):
                good_result+=1
            elif i == 6 and f1_per_class[i]>(0.739-delta):
                good_result+=1
            elif i == 7 and f1_per_class[i]>(0.6646-delta):
                good_result+=1
            elif i == 8 and f1_per_class[i]>(0.5910-delta):
                good_result+=1
            elif i == 9 and f1_per_class[i]>(0.5684-delta):
                good_result+=1
          elif da == "UK":
            if i == 5 and f1_per_class[i]>(0.61-delta):
                good_result+=1
            elif i == 6 and f1_per_class[i]>(0.67-delta):
                good_result+=1
            elif i == 7 and f1_per_class[i]>(0.67-delta):
                good_result+=1
            elif i == 8 and f1_per_class[i]>(0.61-delta):
                good_result+=1
            elif i == 9 and f1_per_class[i]>(0.53-delta):
                good_result+=1
          elif da == "CA":
            if i == 5 and f1_per_class[i]>(0.65-delta):
                good_result+=1
            elif i == 6 and f1_per_class[i]>(0.60-delta):
                good_result+=1
            elif i == 7 and f1_per_class[i]>(0.66-delta):
                good_result+=1
            elif i == 8 and f1_per_class[i]>(0.44-delta):
                good_result+=1
            elif i == 9 and f1_per_class[i]>(0.58-delta):
                good_result+=1
          #if da == "AU":
          #  if i == 5 and f1_per_class[i]>(0.7537-delta):
          #      good_result+=1
          #  elif i == 6 and f1_per_class[i]>(0.739-delta):
          #      good_result+=1
          #  elif i == 7 and f1_per_class[i]>(0.6646-delta):
          #      good_result+=1
          #  elif i == 8 and f1_per_class[i]>(0.5910-delta):
          #      good_result+=1
          #  elif i == 9 and f1_per_class[i]>(0.5684-delta):
          #      good_result+=1
          #elif da == "UK":
          #  if i == 5 and f1_per_class[i]>(0.71-delta):
          #      good_result+=1
          #  elif i == 6 and f1_per_class[i]>(0.67-delta):
          #      good_result+=1
          #  elif i == 7 and f1_per_class[i]>(0.77-delta):
          #      good_result+=1
          #  elif i == 8 and f1_per_class[i]>(0.68-delta):
          #      good_result+=1
          #  elif i == 9 and f1_per_class[i]>(0.54-delta):
          #      good_result+=1
          #elif da == "CA":
          #  if i == 5 and f1_per_class[i]>(0.73-delta):
          #      good_result+=1
          #  elif i == 6 and f1_per_class[i]>(0.68-delta):
          #      good_result+=1
          #  elif i == 7 and f1_per_class[i]>(0.7308-delta):
          #      good_result+=1
          #  elif i == 8 and f1_per_class[i]>(0.5907-delta):
          #      good_result+=1
          #  elif i == 9 and f1_per_class[i]>(0.5896-delta):
          #      good_result+=1

          print("good_result ",good_result, " i ",i) 
          print(f"F1-score for class {list_name[i]}: {f1_per_class[i]:.4f}") 
    #print(f"Macro F1-score: {micro_f1:.4f}")
    #print(f"Macro F1-score: {weighted_f1:.4f}")
    #print(f"Macro F1-score: {macro_f1:.4f}")

    #if good_result >= 2:
    #   timestamp = time.strftime("%Y%m%d-%H%M%S")
    #   filename = f"contrastive_{args.label[0]}_{args.label[1]}_{timestamp}.pth"
    #   torch.save(model.state_dict(), filename)
    #   print("filename ",filename)
    if good_result >=2:
        return 1
    else:
        return 0
    

def main(args):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  epochs = args.epochs
  lr = args.lr
  loss_fn = nn.BCELoss(reduction = "none")
  batch_size = args.batch_size
  max_length = 60

  random_seed = 42
  torch.manual_seed(random_seed)
  random.seed(random_seed)
  np.random.seed(random_seed)

  
  #model_name = 'distilbert-base-uncased'
  #model_name = "roberta-base"
  #checkpoint_path = "./cp/contrastive_pretrained_0.pth"  # Replace with your file path
  #state_dict = torch.load(checkpoint_path, map_location="cpu") 
  #filtered_state_dict = {k[5:]: v for k, v in state_dict.items() if "projection_head" not in k}
  model_name = "answerdotai/ModernBERT-large"
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  with open('../data/documents_train.pkl', 'rb') as file:
      texts = pickle.load(file)
  with open('../data/labels_train.pkl', 'rb') as file:
      labels = pickle.load(file)
  labels = torch.tensor(labels)
  mask = (labels[:, args.label[0]] == 1) | (labels[:, args.label[1]] == 1)
  texts = [texts[i] for i in range(len(texts)) if mask[i]]
  labels = labels[mask]
  train_dataset = MyDataset(texts, tokenizer, max_length, labels)
  print("length of the training dataset is ", len(train_dataset))
  #with open('../data/documents_test.pkl', 'rb') as file:
  #    texts = pickle.load(file)
  #with open('../data/labels_test.pkl', 'rb') as file:
  #    labels = pickle.load(file)
  #labels = np.array(labels)[:, [args.label]]
  df = pd.read_csv('../data/val.csv')
  labels = df.iloc[:,-11:].apply(lambda row: row.tolist(), axis=1).tolist()
  texts = df.sentence.tolist()
  val_dataset = MyDataset(texts, tokenizer, max_length, labels)

  df = pd.read_csv('../data/test.csv')
  labels = df.iloc[:,-11:].apply(lambda row: row.tolist(), axis=1).tolist()
  texts = df.sentence.tolist()
  test_dataset = MyDataset(texts, tokenizer, max_length, labels)

  
  df = pd.read_csv("../data/Canadian.csv", encoding='latin1').iloc[:3658,:]
  texts = df.sentence.tolist()
  labels = df.targets
  labels = [ast.literal_eval(i) for i in labels]
  ca_dataset = MyDataset(texts, tokenizer, max_length, labels)

  df = pd.read_csv("../data/UK.csv", encoding='latin1')
  texts = df.sentence.tolist()
  labels = df.targets
  labels = [ast.literal_eval(i) for i in labels]
  uk_dataset = MyDataset(texts, tokenizer, max_length, labels)
  

  #model_name = 'distilbert-base-uncased' 

  # Randomly split the dataset (with reproducibility)
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
  uk_loader = DataLoader(uk_dataset, batch_size=batch_size, shuffle=False)
  ca_loader = DataLoader(ca_dataset, batch_size=batch_size, shuffle=False)

  # If you are training, here’s a basic training loop structure
  model = SentenceBERTForDownstreamTask(tokenizer, model_name,dropout=args.dropout).to(device)
  
  state_dict = torch.load("./fine-tuned/fine_tuned_class_20250307-115045.pth", map_location=device)  # Adjust for GPU if needed
  model.load_state_dict(state_dict, strict=False)

  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  model.train()  # Set the model to train mode
  train_loss = []
  val_loss = []
  test_loss = []
  for epoch in range(epochs):
      epoch_loss = 0
      step = 0
      for batch, labels in tqdm(train_loader):
          torch.cuda.empty_cache()
          labels = labels.to(device)
          optimizer.zero_grad()
          logits, embeddings = train_batch(batch, model)
          mask = (labels != -1).to(device)
          labels_for_loss = labels.clone().to(device)
          labels_for_loss[~mask] = 0  # Replace -1 with 0 where mask is False
          loss_per_label = loss_fn(logits, labels_for_loss)
          masked_loss = loss_per_label * mask.float()
          loss = masked_loss.sum() / mask.sum()
          #loss = loss_fn(logits, labels).mean()
          #print("loss ",loss)
          cl = contrastive_loss(embeddings, labels, args.label[0], args.label[1])
          epoch_loss += loss.item()*batch_size + 0.01 * cl
          
          #0.05 work for au, 0.03 for 79 0.1 for 56 78

          #loss += 0.01 * cl
          loss += args.lam * cl
          loss.backward()
          optimizer.step()
      
          if step > 2000:
             step = 0
             score = 0
             print("######## AU ######")
             this_score = evaluate(model, test_loader, loss_fn,batch_size,len(test_dataset),"AU")
             score += this_score
             print("######## CA ######")
             this_score = evaluate(model, ca_loader, loss_fn,batch_size,len(ca_dataset), "CA")
             score += this_score
             print("######## UK ######")
             this_score = evaluate(model, uk_loader, loss_fn,batch_size,len(uk_dataset), "UK")
             score += this_score
             if score >2:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = f"contrastive_{args.label[0]}_{args.label[1]}_{timestamp}.pth"
                torch.save(model.state_dict(), filename)
                print("filename ",filename)
          step+=1
      train_loss.append(epoch_loss/len(train_dataset))

      print("#######################VAL")
      this_val_loss, f1 = evaluate(model, val_loader, loss_fn,batch_size, len(val_dataset))
      val_loss.append(this_val_loss)

      print("#######################3TEST")
      this_test_loss,f1 = evaluate(model, test_loader, loss_fn,batch_size,len(test_dataset))
      test_loss.append(this_test_loss)
      print(f"Epoch {epoch+1}/{epochs}, Test Loss: {this_test_loss}")
      print("Test mF1: ", f1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Training Script with Arguments")

    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train the model")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for DataLoader")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate for the model")
    parser.add_argument("--dropout", type=float, default=0.3, help="dropout")
    parser.add_argument('--label', nargs='+', type=int, help='labels for contrastive learning')
    parser.add_argument("--lam", type=float, default=1e-2, help="lambda for contrastive learning")
    args = parser.parse_args()
    print("HYPERPARAMETERS@@@@@@@@@ learning rate",args.lr, " batch size :",args.batch_size," number of epochs ", args.epochs,
         " label ", args.label, " lambda ",args.lam)
    main(args)
