import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from pathlib import Path

# Import model
from context_agent_classifier import ContextAgentClassifier

# Configuration
CSV_FILE = "/Users/aringarg/Downloads/690F_FP/final_project/restbankbig.csv"
MODEL_SAVE_PATH = "context_agent_mlp.pth"
BATCH_SIZE = 8  # Reduced batch size for small dataset
EPOCHS = 15     # Increased epochs slightly since data is small
LEARNING_RATE = 1e-3

class ContextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

def load_data(csv_path):
    """
    Load data from restbankbig.csv
    Expects columns: 'prompt', 'domain'
    """        
    print(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    
    texts = []
    labels = []
    
    # Map string labels to integers
    label_map = {"restaurant": 0, "bank": 1}
    
    for idx, row in df.iterrows():
        text = str(row['prompt'])
        domain = str(row['domain']).lower().strip()

        if domain in label_map:
            texts.append(text)
            labels.append(label_map[domain])
            
    print(f"Found {len(texts)} samples.")
    print(f"Counts: Restaurant (0): {labels.count(0)}, Bank (1): {labels.count(1)}")
    
    return texts, labels

def train():
    # Prepare Data
    try:
        texts, labels = load_data(CSV_FILE)
    except FileNotFoundError:
        print(f"Error: {CSV_FILE} not found.")
        return

    # Split into Train/Val
    # specific random_state ensures that we get a good mix even with small data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    train_dataset = ContextDataset(train_texts, train_labels)
    val_dataset = ContextDataset(val_texts, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Initialize Model
    # Explicitly creating model on CPU to avoid M1/M2 crash
    model = ContextAgentClassifier()
    
    optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    print("\nStarting Training...")
    print("-" * 40)
    
    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_texts, batch_labels in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(list(batch_texts)) 
            
            # Calculate loss
            loss = criterion(logits, batch_labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_texts, batch_labels in val_loader:
                logits = model(list(batch_texts))
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == batch_labels).sum().item()
                total += len(batch_labels)
        
        val_acc = correct / total
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2%}")

    # Save model
    model.save_model(MODEL_SAVE_PATH)

if __name__ == "__main__":
    train()