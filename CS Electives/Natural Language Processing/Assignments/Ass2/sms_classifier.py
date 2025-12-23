#!/usr/bin/env python3
import numpy as np
import pandas as pd
import re
import random
from collections import Counter
from typing import Tuple, Dict, List

# Implement train_test_split from scratch 
def train_test_split(X, y, test_size=0.2, random_state=None, stratify=None):
    
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)
    
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    
    if stratify is not None:
        indices_by_class = {}
        for i, label in enumerate(stratify):
            if label not in indices_by_class:
                indices_by_class[label] = []
            indices_by_class[label].append(i)
        
        test_indices = []
        train_indices = []
        
        for label, indices in indices_by_class.items():
            n_class_test = int(len(indices) * test_size)
            shuffled_indices = indices.copy()
            random.shuffle(shuffled_indices)
            
            test_indices.extend(shuffled_indices[:n_class_test])
            train_indices.extend(shuffled_indices[n_class_test:])
    else:
        indices = list(range(n_samples))
        random.shuffle(indices)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
    
    X_train = [X[i] for i in train_indices]
    X_test = [X[i] for i in test_indices]
    y_train = [y[i] for i in train_indices]
    y_test = [y[i] for i in test_indices]
    
    return X_train, X_test, y_train, y_test


# Implement evaluation metrics from scratch
def accuracy_score(y_true, y_pred):
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct / len(y_true)

# Precision score
def precision_score(y_true, y_pred, pos_label='spam'):
    true_positives = sum(1 for true, pred in zip(y_true, y_pred) 
                        if true == pos_label and pred == pos_label)
    predicted_positives = sum(1 for pred in y_pred if pred == pos_label)
    
    if predicted_positives == 0:
        return 0.0
    return true_positives / predicted_positives

# Recall score
def recall_score(y_true, y_pred, pos_label='spam'):
    true_positives = sum(1 for true, pred in zip(y_true, y_pred) 
                        if true == pos_label and pred == pos_label)
    actual_positives = sum(1 for true in y_true if true == pos_label)
    
    if actual_positives == 0:
        return 0.0
    return true_positives / actual_positives

# F1 score
def f1_score(y_true, y_pred, pos_label='spam'):
    precision = precision_score(y_true, y_pred, pos_label)
    recall = recall_score(y_true, y_pred, pos_label)
    
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

# Confusion matrix
def confusion_matrix(y_true, y_pred, labels=None):
    if labels is None:
        labels = sorted(list(set(y_true + y_pred)))
    
    n_labels = len(labels)
    label_to_idx = {label: i for i, label in enumerate(labels)}
    
    cm = np.zeros((n_labels, n_labels), dtype=int)
    
    for true, pred in zip(y_true, y_pred):
        true_idx = label_to_idx[true]
        pred_idx = label_to_idx[pred]
        cm[true_idx, pred_idx] += 1
    
    return cm

# Text preprocessor
class TextPreprocessor:    
    def __init__(self, lowercase=True, remove_punctuation=True, 
                 remove_numbers=False, min_word_length=1):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.min_word_length = min_word_length

    # Preprocess the text
    def preprocess(self, text: str) -> str:
        if self.lowercase:
            text = text.lower()
            
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
    
        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', ' ', text)
            
        # Remove extra whitespaces and filter by word length
        words = text.split()
        words = [word for word in words if len(word) >= self.min_word_length]
    
        
        return ' '.join(words)

# Naive Bayes classifier
class NaiveBayesClassifier:    
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Laplace smoothing parameter
        self.class_priors = {} # P(c) for each class
        self.feature_likelihoods = {} # P(wᵢ|c) for each word/class pair
        self.vocabulary = set() # All unique words V
        self.classes = []  # [ham, spam]
        self.vocab_size = 0  # |V|
        self._trained = False
        
    def fit(self, X: List[str], y: List[str]) -> None:
        self.classes = list(set(y))
        n_samples = len(X)
        
        # Mathematical Formula: P(c) = count(c) / total_documents
        class_counts = Counter(y)
        for cls in self.classes:
            self.class_priors[cls] = class_counts[cls] / n_samples
            
        # Extract all unique words from training data
        all_words = set()
        for text in X:
            words = text.split()
            all_words.update(words)
        self.vocabulary = all_words
        self.vocab_size = len(self.vocabulary)
        
        # Calculate feature likelihoods for each class
        for cls in self.classes:
            # Get all texts for this class
            class_texts = [X[i] for i in range(len(X)) if y[i] == cls]
            
            # Count word occurrences in this class
            word_counts = Counter()
            total_words = 0
            
            for text in class_texts:
                words = text.split()
                word_counts.update(words)
                total_words += len(words)
            
            # Apply Laplace smoothing
            self.feature_likelihoods[cls] = {}
            for word in self.vocabulary:
                count = word_counts.get(word, 0)
                # Laplace smoothing
                self.feature_likelihoods[cls][word] = (count + self.alpha) / (total_words + self.alpha * self.vocab_size)
        
        self._trained = True
    
    # Calculate log-probabilities to avoid numerical underflow
    def predict_proba(self, X: List[str]) -> np.ndarray:
        if not self._trained:
            raise ValueError("Model must be fitted before making predictions")
            
        probabilities = []
        
        for text in X:
            class_probs = {}
            words = text.split()
            
            for cls in self.classes:
                # Start with log prior: log P(c)
                log_prob = np.log(self.class_priors[cls])
                
                # Add log likelihoods: log P(wᵢ|c)
                for word in words:
                    if word in self.vocabulary:
                        log_prob += np.log(self.feature_likelihoods[cls][word])
                    else:
                        # Handle unseen words with smoothing
                        log_prob += np.log(self.alpha / (sum(len(text.split()) for text in X if text) + self.alpha * self.vocab_size))
                
                class_probs[cls] = log_prob
            
            # Convert back to probabilities and normalize
            max_log_prob = max(class_probs.values())
            probs = {cls: np.exp(log_prob - max_log_prob) for cls, log_prob in class_probs.items()}
            total_prob = sum(probs.values())
            probs = {cls: prob / total_prob for cls, prob in probs.items()}
            
            probabilities.append([probs[cls] for cls in self.classes])
        
        return np.array(probabilities)
    
    # Predict the model
    def predict(self, X: List[str]) -> List[str]:
        probabilities = self.predict_proba(X)
        predictions = []
        
        for prob_row in probabilities:
            predicted_class_idx = np.argmax(prob_row)
            predictions.append(self.classes[predicted_class_idx])
            
        return predictions

# Artificial Neural Network with one hidden layer using ReLU activation
# Code piece taken from Miss Ayesha Enayat slides
class ANNClassifier:
    
    def __init__(self, hidden_size=128, learning_rate=0.01, epochs=100):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self._trained = False

    # Sigmoid activation function for output layer
    def _sigmoid(self, x):
        x = np.clip(x, -500, 500) 
        return 1 / (1 + np.exp(-x))
    
    # ReLU activation function for hidden layer
    def _relu(self, x):
        return np.maximum(0, x)
    
    # ReLU derivative
    def _relu_derivative(self, x):
        return (x > 0).astype(float)
    
    # Initialize weights randomly with small random values
    def _initialize_weights(self, input_size):
        np.random.seed(42)  
        
        self.W1 = np.random.randn(input_size, self.hidden_size) * 0.01
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, 1) * 0.01
        self.b2 = np.zeros((1, 1))
    
    # Forward pass
    def _forward_pass(self, X):
        # Hidden layer with ReLU
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self._relu(self.z1)
        
        # Output layer with sigmoid
        self.a2 = self._sigmoid(np.dot(self.a1, self.W2) + self.b2)
        
        return self.a2
    
    # Backward pass 
    def _backward_pass(self, X, y, output):
        m = X.shape[0]
        
        # Output layer gradients
        dz2 = output - y.reshape(-1, 1)
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # Hidden layer gradients
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self._relu_derivative(self.z1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        return dW1, db1, dW2, db2
    
    # Fit the model
    def fit(self, X, y):
        self._initialize_weights(X.shape[1])
        
        # Convert labels to binary (ham=0, spam=1) for training
        y_binary = np.array([1 if label == 'spam' else 0 for label in y])
        
        # Training loop for epochs
        for epoch in range(self.epochs):
            # Forward pass
            output = self._forward_pass(X)
            
            # Backward pass
            dW1, db1, dW2, db2 = self._backward_pass(X, y_binary, output)
            
            # Update weights
            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1
            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2
        
        self._trained = True
    
    # Predict the model using sigmoid function
    def predict(self, X):
        if not self._trained:
            raise ValueError("Model must be fitted before making predictions")
        
        output = self._forward_pass(X)
        predictions = (output > 0.5).astype(int).flatten()
        return ['spam' if pred == 1 else 'ham' for pred in predictions]

# SMS classifier
class SMSClassifier:

    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.nb_model = None
        self.ann_model = None
        self.label_encoder = {'ham': 0, 'spam': 1}
        self.reverse_label_encoder = {0: 'ham', 1: 'spam'}
    
    # Load the data
    def load_data(self, filepath: str) -> Tuple[List[str], List[str]]:
        texts = []
        labels = []
        
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = line.strip().split('\t', 1)
                if len(parts) == 2:
                    label, text = parts
                    labels.append(label)
                    texts.append(text)
        
        return texts, labels
    
    # Prepare the data for training and testing
    def prepare_data(self, texts: List[str], labels: List[str]) -> Tuple:
        # Preprocess texts
        processed_texts = [self.preprocessor.preprocess(text) for text in texts]
        
        # Create train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            processed_texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
    
        # Report class distribution
        train_counts = Counter(y_train)
        test_counts = Counter(y_test)
        
        print(f"Dataset: {len(texts)} samples | Train: {len(X_train)} | Test: {len(X_test)}")
        print(f"Train - Ham: {train_counts['ham']} ({train_counts['ham']/len(y_train)*100:.1f}%) | Spam: {train_counts['spam']} ({train_counts['spam']/len(y_train)*100:.1f}%)")
        
        return X_train, X_test, y_train, y_test
    
    # Create one-hot features for the ANN model
    def create_one_hot_features(self, texts: List[str], vocabulary: set = None) -> Tuple[np.ndarray, set]:
        if vocabulary is None:
            # Build vocabulary from training data
            vocabulary = set()
            for text in texts:
                vocabulary.update(text.split())
        
        vocab_list = sorted(list(vocabulary))
        word_to_idx = {word: idx for idx, word in enumerate(vocab_list)}
        
        # Create binary feature matrix
        features = np.zeros((len(texts), len(vocabulary)))
        
        for i, text in enumerate(texts):
            words = text.split()
            for word in words:
                if word in word_to_idx:
                    features[i, word_to_idx[word]] = 1
        
        return features, vocabulary
    
    # Train the naive bayes model 
    def train_naive_bayes(self, X_train: List[str], y_train: List[str], 
                         X_val: List[str], y_val: List[str]) -> Dict:
        # Hyperparameter tuning for alpha
        alpha_values = np.logspace(-2, 1, 10)
    
        best_alpha = None
        best_score = 0
        results = {}
        
        for alpha in alpha_values:
            nb = NaiveBayesClassifier(alpha=alpha)
            nb.fit(X_train, y_train)
            
            # Validate
            val_predictions = nb.predict(X_val)
            val_score = accuracy_score(y_val, val_predictions)
            results[alpha] = val_score
            
            if val_score > best_score:
                best_score = val_score
                best_alpha = alpha
        
        print(f"Best alpha: {best_alpha} (Validation Accuracy: {best_score:.4f})")
        
        # Train final model with best alpha
        self.nb_model = NaiveBayesClassifier(alpha=best_alpha)
        self.nb_model.fit(X_train, y_train)
        
        # Evaluate final model on validation set with all metrics
        val_predictions = self.nb_model.predict(X_val)
        val_metrics = {
            'accuracy': accuracy_score(y_val, val_predictions),
            'precision': precision_score(y_val, val_predictions, pos_label='spam'),
            'recall': recall_score(y_val, val_predictions, pos_label='spam'),
            'f1': f1_score(y_val, val_predictions, pos_label='spam')
        }
        
        print(f"\nNAIVE BAYES Validation Results:")
        print(f"Accuracy: {val_metrics['accuracy']:.4f} | Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f} | F1: {val_metrics['f1']:.4f}")
        
        return {'best_alpha': best_alpha, 'validation_results': results, 'validation_metrics': val_metrics}
    
    def train_ann(self, X_train, y_train, X_val, y_val):
        # Try different hyperparameters with better learning rates
        configs = [
            {'hidden_size': 64, 'learning_rate': 0.1, 'epochs': 100},    # Higher LR
            {'hidden_size': 128, 'learning_rate': 0.05, 'epochs': 100},  # Higher LR
            {'hidden_size': 64, 'learning_rate': 0.01, 'epochs': 150},   # More epochs
            {'hidden_size': 32, 'learning_rate': 1, 'epochs': 100},    # Smaller network
        ]
        
        best_config = None
        best_score = 0
        
        print("Tuning ANN hyperparameters...")
        for i, config in enumerate(configs):
            print(f"\nConfig {i+1}: Hidden={config['hidden_size']}, LR={config['learning_rate']}, Epochs={config['epochs']}")
            
            ann = ANNClassifier(**config)
            ann.fit(X_train, y_train)
            
            # Validate
            val_predictions = ann.predict(X_val)
            val_score = accuracy_score(y_val, val_predictions)
            
            # Check if it's predicting both classes
            spam_count = sum(1 for pred in val_predictions if pred == 'spam')
            print(f"Validation Accuracy: {val_score:.4f} | Spam predictions: {spam_count}/{len(val_predictions)}")
            
            if val_score > best_score:
                best_score = val_score
                best_config = config
        
        print(f"\nBest ANN config: Hidden={best_config['hidden_size']}, LR={best_config['learning_rate']} (Validation Accuracy: {best_score:.4f})")
        
        # Train final model with best configuration
        print("\nTraining final ANN model...")
        self.ann_model = ANNClassifier(**best_config)
        self.ann_model.fit(X_train, y_train)
        
        # Evaluate final model on validation set with all metrics
        val_predictions = self.ann_model.predict(X_val)
        val_metrics = {
            'accuracy': accuracy_score(y_val, val_predictions),
            'precision': precision_score(y_val, val_predictions, pos_label='spam'),
            'recall': recall_score(y_val, val_predictions, pos_label='spam'),
            'f1': f1_score(y_val, val_predictions, pos_label='spam')
        }
        
        print(f"\nANN Validation Results:")
        print(f"Accuracy: {val_metrics['accuracy']:.4f} | Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f} | F1: {val_metrics['f1']:.4f}")
        
        return {'best_config': best_config, 'validation_accuracy': best_score, 'validation_metrics': val_metrics}
    
    def evaluate_model(self, model, X_test, y_test, model_name: str):
        """Evaluate Naive Bayes model and return metrics"""
        # Get predictions from Naive Bayes model
        pred_labels = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, pred_labels)
        precision = precision_score(y_test, pred_labels, pos_label='spam')
        recall = recall_score(y_test, pred_labels, pos_label='spam')
        f1 = f1_score(y_test, pred_labels, pos_label='spam')
        
        print(f"\n{model_name} Test Results:")
        print("-" * 30)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, pred_labels, labels=['ham', 'spam'])
        print(f"\nConfusion Matrix:")
        print(f"              Predicted")
        print(f"           Ham    Spam")
        print(f"Actual Ham  {cm[0,0]:4d}   {cm[0,1]:4d}")
        print(f"     Spam   {cm[1,0]:4d}   {cm[1,1]:4d}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'predictions': pred_labels
        }
    
    def compare_models(self, nb_results: Dict, ann_results: Dict, nb_training: Dict, ann_training: Dict):
        """Fair comparison and analysis between NB and ANN"""
        print(f"\n" + "="*50)
        print("D. FAIR COMPARISON & ANALYSIS")
        print("="*50)
        
        # Validation Results Table
        print(f"\nValidation Performance:")
        print(f"{'Metric':<12} {'Naive Bayes':<12} {'ANN':<12} {'Winner':<10}")
        print("-" * 50)
        
        val_metrics = ['accuracy', 'precision', 'recall', 'f1']
        nb_val = nb_training.get('validation_metrics', {})
        ann_val = ann_training.get('validation_metrics', {})
        
        for metric in val_metrics:
            nb_val_score = nb_val.get(metric, 0.0)
            ann_val_score = ann_val.get(metric, 0.0)
            
            if nb_val_score > ann_val_score:
                winner = "NB"
            elif ann_val_score > nb_val_score:
                winner = "ANN"
            else:
                winner = "Tie"
                
            print(f"{metric.capitalize():<12} {nb_val_score:<12.4f} {ann_val_score:<12.4f} {winner:<10}")
        
        # Test Results Table
        print(f"\nTest Performance:")
        print(f"{'Metric':<12} {'Naive Bayes':<12} {'ANN':<12} {'Winner':<10}")
        print("-" * 50)
        
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        winners = {'nb': 0, 'ann': 0, 'tie': 0}
        
        for metric in metrics:
            nb_val = nb_results[metric]
            ann_val = ann_results[metric]
            
            if nb_val > ann_val:
                winner = "NB"
                winners['nb'] += 1
            elif ann_val > nb_val:
                winner = "ANN"
                winners['ann'] += 1
            else:
                winner = "Tie"
                winners['tie'] += 1
                
            print(f"{metric.capitalize():<12} {nb_val:<12.4f} {ann_val:<12.4f} {winner:<10}")
        
        return {
            'winners': winners
        }
    
    def run_experiment(self):
        print("SMS SPAM CLASSIFICATION - COMPLETE EXPERIMENT")
        print("="*60)
        
        # A. DATA PREPARATION
        print("\nA. DATA PREPARATION")
        print("-" * 30)
        texts, labels = self.load_data('SMSSpamCollection')
        X_train, X_test, y_train, y_test = self.prepare_data(texts, labels)

        # Create validation split from training data for hyperparameter tuning
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # A.2: Prepare one-hot features (required for ANN)
        X_train_onehot, vocabulary = self.create_one_hot_features(X_train_split)
        X_val_onehot, _ = self.create_one_hot_features(X_val, vocabulary)
        X_test_onehot, _ = self.create_one_hot_features(X_test, vocabulary)
        print(f"Features: {X_train_onehot.shape[1]}")
        
        # B. NAIVE BAYES IMPLEMENTATION 
        print("\nB. NAIVE BAYES CLASSIFIER")
        print("-" * 30)
        nb_results = self.train_naive_bayes(X_train_split, y_train_split, X_val, y_val)
        nb_test_results = self.evaluate_model(self.nb_model, X_test, y_test, "NAIVE BAYES")
        
        # C. ANN IMPLEMENTATION
        print("\nC. ANN WITH ONE HIDDEN LAYER")
        print("-" * 30)
        ann_results = self.train_ann(X_train_onehot, y_train_split, X_val_onehot, y_val)
        ann_test_results = self.evaluate_model(self.ann_model, X_test_onehot, y_test, "ANN")
        
        # D. FAIR COMPARISON & ANALYSIS
        self.compare_models(nb_test_results, ann_test_results, nb_results, ann_results)
        
        return {
            'nb_results': nb_test_results,
            'ann_results': ann_test_results,
            'nb_training': nb_results,
            'ann_training': ann_results,
            'vocabulary_size': len(vocabulary)
        }
    
def main():
    """Main execution function"""
    classifier = SMSClassifier()
    results = classifier.run_experiment()
    
    
    return results

if __name__ == "__main__":
    main()