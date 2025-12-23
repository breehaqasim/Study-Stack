
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import re
from collections import Counter
import unicodedata
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ============================================================================
# PART 1: DATA LOADING AND PREPROCESSING
# ============================================================================

class UrduTextPreprocessor:
    """
    Urdu text preprocessing utilities
    """
    
    def __init__(self):
        # Urdu Unicode range: 0600-06FF (Arabic/Urdu block)
        self.urdu_pattern = re.compile(r'[\u0600-\u06FF]+')
        
    def normalize_urdu(self, text):
        """Normalize Urdu text"""
        if not isinstance(text, str):
            return ""
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Normalize Arabic/Urdu characters
        text = unicodedata.normalize('NFKC', text)
        
        # Remove English characters and numbers
        text = re.sub(r'[a-zA-Z0-9]', '', text)
        
        # Remove punctuation except Urdu punctuation
        text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
        
        # Remove extra spaces
        text = ' '.join(text.split())
        
        return text.strip()
    
    def tokenize(self, text):
        """Simple whitespace tokenization for Urdu"""
        return text.split()
    
    def preprocess(self, text):
        """Complete preprocessing pipeline"""
        text = self.normalize_urdu(text)
        tokens = self.tokenize(text)
        return tokens


def load_urdu_dataset(file_path):
    """
    Load Urdu Sentiment Corpus
    Expected format: CSV/TSV with columns ['text', 'label'] or similar
    
    Args:
        file_path: Path to dataset file
        
    Returns:
        texts: List of text samples
        labels: List of sentiment labels
    """
    print("="*80)
    print("Loading Urdu Sentiment Dataset")
    print("="*80)
    
    try:
        # Load TSV file (tab-separated)
        print("Loading TSV format (tab-separated)")
        df = pd.read_csv(file_path, encoding='utf-8', sep='\t')
        
        # Common column name variations
        text_col = None
        label_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if any(name in col_lower for name in ['text', 'review', 'comment', 'sentence']):
                text_col = col
            if any(name in col_lower for name in ['label', 'sentiment', 'class']):
                label_col = col
        
        if text_col is None or label_col is None:
            print(f"Available columns: {df.columns.tolist()}")
            print("Using first two columns as [text, label]")
            text_col = df.columns[0]
            label_col = df.columns[1]
        
        # Drop rows with missing values
        df = df.dropna(subset=[text_col, label_col])
        
        # Convert to lists and ensure labels are strings
        texts = df[text_col].astype(str).tolist()
        labels = df[label_col].astype(str).tolist()
        
        print(f"✓ Loaded {len(texts)} samples")
        print(f"✓ Text column: '{text_col}'")
        print(f"✓ Label column: '{label_col}'")
        
        # Analyze labels
        unique_labels = sorted(set(labels))
        print(f"✓ Unique labels: {unique_labels}")
        
        label_counts = Counter(labels)
        for label, count in label_counts.items():
            print(f"  - {label}: {count} samples ({count/len(labels)*100:.1f}%)")
        
        return texts, labels
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nPlease ensure:")
        print("1. Dataset file exists")
        print("2. File format is CSV")
        print("3. Contains 'text' and 'label' columns")
        raise


class Vocabulary:
    """
    Vocabulary builder for Urdu text
    """
    
    def __init__(self, max_vocab_size=10000, min_freq=2):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.word_freq = Counter()
        
    def build_vocab(self, tokenized_texts):
        """Build vocabulary from tokenized texts"""
        print("\n" + "="*80)
        print("Building Vocabulary")
        print("="*80)
        
        # Count word frequencies
        for tokens in tokenized_texts:
            self.word_freq.update(tokens)
        
        # Filter by frequency and limit size
        sorted_words = sorted(self.word_freq.items(), key=lambda x: x[1], reverse=True)
        vocab_words = [word for word, freq in sorted_words 
                      if freq >= self.min_freq][:self.max_vocab_size - 2]
        
        # Create mappings
        for idx, word in enumerate(vocab_words, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        
        print(f"✓ Vocabulary size: {len(self.word2idx)}")
        print(f"✓ Total unique words: {len(self.word_freq)}")
        print(f"✓ Min frequency: {self.min_freq}")
        
        return self
    
    def encode(self, tokens):
        """Convert tokens to indices"""
        return [self.word2idx.get(token, 1) for token in tokens]  # 1 = <UNK>
    
    def decode(self, indices):
        """Convert indices to tokens"""
        return [self.idx2word.get(idx, '<UNK>') for idx in indices]


class UrduSentimentDataset(Dataset):
    """
    PyTorch Dataset for Urdu Sentiment Classification
    """
    
    def __init__(self, texts, labels, vocab, preprocessor, max_length=100):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.preprocessor = preprocessor
        self.max_length = max_length
        
        # Convert labels to numerical
        self.label2idx = {label: idx for idx, label in enumerate(sorted(set(labels)))}
        self.idx2label = {idx: label for label, idx in self.label2idx.items()}
        self.num_classes = len(self.label2idx)
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        # Preprocess text
        tokens = self.preprocessor.preprocess(self.texts[idx])
        
        # Encode tokens
        encoded = self.vocab.encode(tokens)
        
        # Pad or truncate
        if len(encoded) > self.max_length:
            encoded = encoded[:self.max_length]
        else:
            encoded = encoded + [0] * (self.max_length - len(encoded))
        
        # Convert label
        label = self.label2idx[self.labels[idx]]
        
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long)


# ============================================================================
# PART 2: BiLSTM MODEL
# ============================================================================

class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM for sentiment classification
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes,
                 num_layers=2, dropout=0.3):
        super(BiLSTMClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                             batch_first=True, bidirectional=True,
                             dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional
        
    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        
        # BiLSTM
        output, (hidden, cell) = self.bilstm(embedded)
        
        # Use last hidden state (concatenate forward and backward)
        last_hidden = output[:, -1, :]  # Already concatenated by PyTorch
        last_hidden = self.dropout(last_hidden)
        
        logits = self.fc(last_hidden)
        
        return logits


# ============================================================================
# PART 3: TRAINING AND EVALUATION
# ============================================================================

class Trainer:
    """
    Training and evaluation utilities
    """
    
    def __init__(self, model, train_loader, val_loader, criterion, optimizer,
                 device, num_classes):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.num_classes = num_classes
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        
        return avg_loss, accuracy, precision, recall, f1
    
    def train(self, num_epochs):
        """Complete training loop"""
        print("\n" + "="*80)
        print("Training Model")
        print("="*80)
        
        best_f1 = 0
        best_model_state = None
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 80)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # Validate
            val_loss, val_acc, val_prec, val_rec, val_f1 = self.validate()
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
            print(f"Val Precision: {val_prec:.4f} | Val Recall: {val_rec:.4f} | Val F1: {val_f1:.4f}")
            
            # Save best model
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_model_state = self.model.state_dict().copy()
                print(f"✓ New best F1: {best_f1:.4f}")
        
        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            print(f"\n✓ Loaded best model with F1: {best_f1:.4f}")
        
        return best_f1


def evaluate_model(model, test_loader, dataset, device):
    """
    Comprehensive evaluation on test set
    """
    print("\n" + "="*80)
    print("FINAL EVALUATION ON TEST SET")
    print("="*80)
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    print("\n📊 Overall Metrics:")
    print("-" * 80)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # Per-class metrics
    print("\n📋 Classification Report:")
    print("-" * 80)
    target_names = [dataset.idx2label[i] for i in range(dataset.num_classes)]
    print(classification_report(all_labels, all_preds, target_names=target_names, zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix', fontsize=14, pad=15)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Confusion matrix saved to confusion_matrix.png")
    plt.close()
    
    return accuracy, precision, recall, f1


# ============================================================================
# PART 4: MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution pipeline
    """
    print("="*80)
    print("URDU SENTIMENT CLASSIFICATION WITH BiLSTM")
    print("NLP Assignment 3 - Question 2")
    print("="*80)
    
    # ========================
    # CONFIGURATION
    # ========================
    CONFIG = {
        'dataset_path': 'urdu-sentiment-corpus-v1.tsv',
        'max_vocab_size': 10000,
        'min_word_freq': 2,
        'max_length': 100,
        'embedding_dim': 128,
        'hidden_dim': 256,
        'num_layers': 2,
        'dropout': 0.2,
        'batch_size': 32,
        'learning_rate': 0.0001,
        'num_epochs': 30
    }
    
    print("\n📋 Configuration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    # ========================
    # LOAD DATA
    # ========================
    texts, labels = load_urdu_dataset(CONFIG['dataset_path'])
    
    # ========================
    # PREPROCESSING
    # ========================
    print("\n" + "="*80)
    print("Preprocessing Text")
    print("="*80)
    
    preprocessor = UrduTextPreprocessor()
    tokenized_texts = [preprocessor.preprocess(text) for text in tqdm(texts, desc="Tokenizing")]
    
    # Remove empty samples
    valid_indices = [i for i, tokens in enumerate(tokenized_texts) if len(tokens) > 0]
    tokenized_texts = [tokenized_texts[i] for i in valid_indices]
    labels = [labels[i] for i in valid_indices]
    texts = [texts[i] for i in valid_indices]
    
    print(f"✓ Valid samples after preprocessing: {len(tokenized_texts)}")
    
    # Build vocabulary
    vocab = Vocabulary(max_vocab_size=CONFIG['max_vocab_size'], 
                      min_freq=CONFIG['min_word_freq'])
    vocab.build_vocab(tokenized_texts)
    
    # ========================
    # SPLIT DATA
    # ========================
    print("\n" + "="*80)
    print("Splitting Data")
    print("="*80)
    
    # Split: 70% train, 15% validation, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Train: {len(X_train)} samples")
    print(f"Val:   {len(X_val)} samples")
    print(f"Test:  {len(X_test)} samples")
    
    # Create datasets
    train_dataset = UrduSentimentDataset(X_train, y_train, vocab, preprocessor, 
                                         CONFIG['max_length'])
    val_dataset = UrduSentimentDataset(X_val, y_val, vocab, preprocessor,
                                       CONFIG['max_length'])
    test_dataset = UrduSentimentDataset(X_test, y_test, vocab, preprocessor,
                                        CONFIG['max_length'])
    
    print(f"Number of classes: {train_dataset.num_classes}")
    print(f"Label mapping: {train_dataset.label2idx}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'],
                           shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'],
                            shuffle=False, num_workers=0)
    
    # ========================
    # BUILD MODEL
    # ========================
    print("\n" + "="*80)
    print("Building BiLSTM Model")
    print("="*80)
    
    vocab_size = len(vocab.word2idx)
    num_classes = train_dataset.num_classes
    
    model = BiLSTMClassifier(vocab_size, CONFIG['embedding_dim'], CONFIG['hidden_dim'],
                            num_classes, CONFIG['num_layers'], CONFIG['dropout'])
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model: BiLSTM")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # ========================
    # TRAIN MODEL
    # ========================
    
    # Calculate class weights to handle imbalance (fixes low precision!)
    print("\n📊 Calculating class weights for balanced training...")
    train_labels_list = [train_dataset.labels[i] for i in range(len(train_dataset))]
    label_counts_dict = Counter(train_labels_list)
    
    # Compute balanced weights
    class_weights = []
    for i in range(num_classes):
        label_name = train_dataset.idx2label[i]
        count = label_counts_dict.get(label_name, 1)
        weight = len(train_labels_list) / (num_classes * count)
        class_weights.append(weight)
        print(f"  Class '{label_name}': {count} samples → weight: {weight:.3f}")
    
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer,
                     device, num_classes)
    best_f1 = trainer.train(CONFIG['num_epochs'])
    
    # ========================
    # EVALUATE ON TEST SET
    # ========================
    accuracy, precision, recall, f1 = evaluate_model(model, test_loader, 
                                                      test_dataset, device)
    
    # ========================
    # SAVE RESULTS
    # ========================
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"\nModel: BiLSTM")
    print(f"Test Accuracy:  {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall:    {recall:.4f}")
    print(f"Test F1-Score:  {f1:.4f}")
    print("\n" + "="*80)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': CONFIG,
        'vocab': vocab,
        'label_mapping': train_dataset.label2idx,
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    }, 'urdu_sentiment_model.pth')
    
    print("✓ Model saved to urdu_sentiment_model.pth")
    print("\n🏆 Training Complete!")
    

if __name__ == "__main__":
    main()

