"""
Word2Vec Skip-Gram Implementation with Full Softmax
NLP Assignment 3 - Question 1

This script implements:
1. Skip-Gram model with full softmax (no negative sampling)
2. t-SNE visualization of embeddings
3. Unigram and bigram frequency analysis
4. Most similar words functionality
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pickle
import re
from sklearn.manifold import TSNE
from tqdm import tqdm

# Set random seeds for reproducibility
np.random.seed(42)

class SkipGramWord2Vec:
    """
    Skip-Gram Word2Vec implementation with full softmax
    """
    
    def __init__(self, vocab_size, embedding_dim=50, learning_rate=0.01):
        """
        Initialize the Skip-Gram model
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            learning_rate: Learning rate for optimization
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        
        # Initialize weights randomly
        # W_in: input weight matrix (vocab_size x embedding_dim)
        # W_out: output weight matrix (embedding_dim x vocab_size)
        self.W_in = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.W_out = np.random.randn(embedding_dim, vocab_size) * 0.01
        
        # For tracking loss
        self.losses = []
        
    def softmax(self, x):
        """Compute softmax with numerical stability"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def forward(self, center_word_idx):
        """
        Forward pass: compute probabilities for all context words
        
        Args:
            center_word_idx: Index of center word
            
        Returns:
            h: Hidden layer (embedding of center word)
            probs: Probability distribution over vocabulary
        """
        # Get embedding of center word (hidden layer)
        h = self.W_in[center_word_idx]  # Shape: (embedding_dim,)
        
        # Compute scores for all words
        scores = np.dot(h, self.W_out)  # Shape: (vocab_size,)
        
        # Apply softmax to get probabilities
        probs = self.softmax(scores)
        
        return h, probs
    
    def backward(self, center_word_idx, context_word_idx, h, probs):
        """
        Backward pass: compute gradients and update weights
        
        Args:
            center_word_idx: Index of center word
            context_word_idx: Index of context word
            h: Hidden layer from forward pass
            probs: Probabilities from forward pass
        """
        # Compute error signal
        error = probs.copy()
        error[context_word_idx] -= 1  # Subtract 1 from the true context word
        
        # Gradients for output weights
        dW_out = np.outer(h, error)  # Shape: (embedding_dim, vocab_size)
        
        # Gradient for input weights
        dW_in = np.dot(self.W_out, error)  # Shape: (embedding_dim,)
        
        # Update weights using gradient descent
        self.W_out -= self.learning_rate * dW_out
        self.W_in[center_word_idx] -= self.learning_rate * dW_in
        
        # Return loss (negative log likelihood)
        loss = -np.log(probs[context_word_idx] + 1e-10)
        return loss
    
    def train_step(self, center_word_idx, context_word_idx):
        """
        Single training step
        
        Args:
            center_word_idx: Index of center word
            context_word_idx: Index of context word
            
        Returns:
            loss: Cross-entropy loss for this pair
        """
        h, probs = self.forward(center_word_idx)
        loss = self.backward(center_word_idx, context_word_idx, h, probs)
        return loss
    
    def get_embedding(self, word_idx):
        """Get embedding for a word"""
        return self.W_in[word_idx]
    
    def get_all_embeddings(self):
        """Get all word embeddings"""
        return self.W_in


class Word2VecTrainer:
    """
    Trainer class for Word2Vec model
    """
    
    def __init__(self, corpus_path, window_size=2, min_count=1):
        """
        Initialize trainer
        
        Args:
            corpus_path: Path to corpus file
            window_size: Context window size
            min_count: Minimum word frequency to include in vocabulary
        """
        self.corpus_path = corpus_path
        self.window_size = window_size
        self.min_count = min_count
        
        # Vocabulary and mappings
        self.word2idx = {}
        self.idx2word = {}
        self.word_counts = Counter()
        
        # Training data
        self.training_pairs = []
        
        # Load and preprocess corpus
        self.preprocess()
        
    def preprocess(self):
        """
        Preprocess corpus: tokenization, vocabulary building, word-index mapping
        """
        print("Preprocessing corpus...")
        
        # Read corpus
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            text = f.read().lower()
        
        # Tokenization: split by whitespace and basic punctuation
        tokens = re.findall(r'\b\w+\b', text)
        
        # Count word frequencies
        self.word_counts = Counter(tokens)
        
        # Filter by minimum count and build vocabulary
        vocab = [word for word, count in self.word_counts.items() 
                if count >= self.min_count]
        vocab = sorted(vocab)  # Sort for consistency
        
        # Create word-index mappings
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
        print(f"Vocabulary size: {len(self.word2idx)}")
        print(f"Total tokens: {len(tokens)}")
        
        # Convert tokens to indices
        self.token_indices = [self.word2idx[token] for token in tokens 
                             if token in self.word2idx]
        
    def generate_training_pairs(self):
        """
        Generate (center, context) word pairs with sliding window
        """
        print(f"Generating training pairs with window size {self.window_size}...")
        
        self.training_pairs = []
        
        for i, center_idx in enumerate(self.token_indices):
            # Get context window
            start = max(0, i - self.window_size)
            end = min(len(self.token_indices), i + self.window_size + 1)
            
            for j in range(start, end):
                if j != i:  # Skip the center word itself
                    context_idx = self.token_indices[j]
                    self.training_pairs.append((center_idx, context_idx))
        
        print(f"Generated {len(self.training_pairs)} training pairs")
        
    def train(self, embedding_dim=50, learning_rate=0.01, epochs=10, 
              batch_log_interval=1000):
        """
        Train the Skip-Gram model
        
        Args:
            embedding_dim: Dimension of embeddings
            learning_rate: Learning rate
            epochs: Number of training epochs
            batch_log_interval: Log loss every N batches
            
        Returns:
            model: Trained Skip-Gram model
        """
        # Generate training pairs
        self.generate_training_pairs()
        
        # Initialize model
        model = SkipGramWord2Vec(
            vocab_size=len(self.word2idx),
            embedding_dim=embedding_dim,
            learning_rate=learning_rate
        )
        
        print(f"\nTraining Skip-Gram model...")
        print(f"Parameters: embedding_dim={embedding_dim}, lr={learning_rate}, "
              f"epochs={epochs}, window_size={self.window_size}")
        
        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0
            batch_loss = 0
            
            # Shuffle training pairs
            np.random.shuffle(self.training_pairs)
            
            # Progress bar
            pbar = tqdm(self.training_pairs, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch_idx, (center_idx, context_idx) in enumerate(pbar):
                loss = model.train_step(center_idx, context_idx)
                epoch_loss += loss
                batch_loss += loss
                
                # Log batch loss
                if (batch_idx + 1) % batch_log_interval == 0:
                    avg_batch_loss = batch_loss / batch_log_interval
                    pbar.set_postfix({'loss': f'{avg_batch_loss:.4f}'})
                    batch_loss = 0
            
            # Average loss for epoch
            avg_epoch_loss = epoch_loss / len(self.training_pairs)
            model.losses.append(avg_epoch_loss)
            print(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_epoch_loss:.4f}")
        
        return model
    
    def get_word_frequencies(self):
        """Get word frequency distribution"""
        return self.word_counts
    
    def get_bigram_frequencies(self):
        """Compute bigram frequency distribution"""
        bigrams = []
        for i in range(len(self.token_indices) - 1):
            word1 = self.idx2word[self.token_indices[i]]
            word2 = self.idx2word[self.token_indices[i + 1]]
            bigrams.append((word1, word2))
        return Counter(bigrams)


class Word2VecAnalyzer:
    """
    Analyzer class for Word2Vec embeddings
    """
    
    def __init__(self, model, word2idx, idx2word):
        """
        Initialize analyzer
        
        Args:
            model: Trained Skip-Gram model
            word2idx: Word to index mapping
            idx2word: Index to word mapping
        """
        self.model = model
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.embeddings = model.get_all_embeddings()
        
    def cosine_similarity(self, vec1, vec2):
        """Compute cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot_product / (norm1 * norm2 + 1e-10)
    
    def most_similar(self, word, top_k=10):
        """
        Find most similar words to given word using cosine similarity
        
        Args:
            word: Query word
            top_k: Number of similar words to return
            
        Returns:
            List of (word, similarity) tuples
        """
        if word not in self.word2idx:
            print(f"Word '{word}' not in vocabulary")
            return []
        
        # Get embedding of query word
        word_idx = self.word2idx[word]
        word_embedding = self.embeddings[word_idx]
        
        # Compute similarities with all words
        similarities = []
        for idx, other_embedding in enumerate(self.embeddings):
            if idx != word_idx:  # Skip the word itself
                sim = self.cosine_similarity(word_embedding, other_embedding)
                similarities.append((self.idx2word[idx], sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def visualize_tsne(self, perplexity=30, max_iter=1000, save_path='tsne_visualization.png'):
        """
        Visualize embeddings using t-SNE
        
        Args:
            perplexity: t-SNE perplexity parameter
            max_iter: Number of iterations
            save_path: Path to save plot
        """
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=max_iter, 
                   random_state=42)
        embeddings_2d = tsne.fit_transform(self.embeddings)
        
        # Create plot
        plt.figure(figsize=(16, 12))
        
        # Plot points
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                   alpha=0.6, s=100, c='steelblue', edgecolors='black', linewidth=0.5)
        
        # Add labels for all words
        for idx, word in self.idx2word.items():
            plt.annotate(word, 
                        xy=(embeddings_2d[idx, 0], embeddings_2d[idx, 1]),
                        xytext=(3, 3), textcoords='offset points',
                        fontsize=9, alpha=0.8)
        
        plt.title('t-SNE Visualization of Word2Vec Embeddings', fontsize=16, pad=20)
        plt.xlabel('t-SNE Dimension 1', fontsize=12)
        plt.ylabel('t-SNE Dimension 2', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_unigram_distribution(self, word_counts, save_path='unigram_distribution.png'):
        """
        Plot unigram frequency distribution
        
        Args:
            word_counts: Counter object with word frequencies
            save_path: Path to save plot
        """
        plt.figure(figsize=(10, 8))
        
        # Top 20 unigrams
        top_words, top_counts = zip(*word_counts.most_common(20))
        plt.barh(range(len(top_words)), top_counts, color='steelblue', alpha=0.7)
        plt.yticks(range(len(top_words)), top_words)
        plt.gca().invert_yaxis()
        plt.title('Top 20 Most Frequent Words (Unigrams)', fontsize=14, pad=10)
        plt.xlabel('Frequency', fontsize=11)
        plt.ylabel('Words', fontsize=11)
        plt.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Unigram distribution saved to {save_path}")
        plt.close()
    
    def plot_bigram_distribution(self, bigram_counts, save_path='bigram_distribution.png'):
        """
        Plot bigram frequency distribution
        
        Args:
            bigram_counts: Counter object with bigram frequencies
            save_path: Path to save plot
        """
        plt.figure(figsize=(10, 10))
        
        # Top 30 bigrams
        top_bigrams, bigram_freqs = zip(*bigram_counts.most_common(30))
        bigram_labels = [f"{b[0]} {b[1]}" for b in top_bigrams]
        plt.barh(range(len(bigram_labels)), bigram_freqs, color='mediumseagreen', alpha=0.7)
        plt.yticks(range(len(bigram_labels)), bigram_labels, fontsize=9)
        plt.gca().invert_yaxis()
        plt.title('Top 30 Most Frequent Bigrams', fontsize=14, pad=10)
        plt.xlabel('Frequency', fontsize=11)
        plt.ylabel('Bigrams', fontsize=11)
        plt.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Bigram distribution saved to {save_path}")
        plt.close()
    
    def plot_loss_curve(self, save_path='loss_curve.png'):
        """Plot training loss curve"""
        if not self.model.losses:
            print("No loss history available")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.model.losses) + 1), self.model.losses, 
                marker='o', linewidth=2, markersize=6, color='darkblue')
        plt.title('Training Loss Curve', fontsize=14, pad=15)
        plt.xlabel('Epoch', fontsize=11)
        plt.ylabel('Average Loss', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Loss curve saved to {save_path}")
        plt.close()


def main():
    
    # Configuration
    CORPUS_PATH = 'corpus.txt'
    WINDOW_SIZE = 2
    EMBEDDING_DIM = 50
    LEARNING_RATE = 0.01
    EPOCHS = 10
    
    # Initialize trainer
    trainer = Word2VecTrainer(
        corpus_path=CORPUS_PATH,
        window_size=WINDOW_SIZE,
        min_count=1
    )
    
    # Train model
    model = trainer.train(
        embedding_dim=EMBEDDING_DIM,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS
    )
    
    # Save model
    with open('skipgram_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    # print("\nsaved to skipgram_model.pkl")
    
    # Save vocabulary
    vocab_data = {
        'word2idx': trainer.word2idx,
        'idx2word': trainer.idx2word,
        'word_counts': trainer.word_counts
    }
    with open('vocabulary.pkl', 'wb') as f:
        pickle.dump(vocab_data, f)
    # print("saved to vocabulary.pkl")
    
    # Analysis and visualization
    analyzer = Word2VecAnalyzer(model, trainer.word2idx, trainer.idx2word)
    
    print("\n" + "="*60)
    print("Analysis and Visualization")
    print("="*60)
    
    # 1. t-SNE visualization
    print("\nGenerating t-SNE visualization...")
    analyzer.visualize_tsne(save_path='tsne_visualization.png')
    print("✓ t-SNE visualization saved to tsne_visualization.png")
    
    # 2. Frequency distributions - Separate plots
    word_counts = trainer.get_word_frequencies()
    bigram_counts = trainer.get_bigram_frequencies()
    
    print("\nGenerating frequency distribution plots...")
    analyzer.plot_unigram_distribution(word_counts, save_path='unigram_distribution.png')
    analyzer.plot_bigram_distribution(bigram_counts, save_path='bigram_distribution.png')
    
    # 3. Loss curve
    print("\nGenerating loss curve...")
    analyzer.plot_loss_curve(save_path='loss_curve.png')
    
    # 4. Most similar words examples
    print("\n" + "="*60)
    print("Most Similar Words (Cosine Similarity)")
    print("="*60)
    
    test_words = ['doctor', 'student', 'model', 'city', 'health']
    for word in test_words:
        if word in trainer.word2idx:
            print(f"\nMost similar to '{word}':")
            similar = analyzer.most_similar(word, top_k=5)
            for similar_word, similarity in similar:
                print(f"  {similar_word}: {similarity:.4f}")
    
    print("\n" + "="*60)
    print("✓ Training and analysis complete!")
    print("="*60)

    
if __name__ == "__main__":
    main()