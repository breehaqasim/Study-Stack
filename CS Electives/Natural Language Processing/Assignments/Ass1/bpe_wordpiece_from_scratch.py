#!/usr/bin/env python3
import re
import collections
from typing import Dict, List, Tuple, Set

# read the text file
def read_text_file(filepath: str) -> str:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # normalize whitespace, split on punctuation, and convert to lowercase
        text = text.lower()  # convert to lowercase
        text = re.sub(r'\s+', ' ', text.strip()) 
        return text
        
    except FileNotFoundError:
        print(f"Error: File {filepath} not found")
        return ""
    except Exception as e:
        print(f"Error reading file: {e}")
        return ""

# tokenization function based on updated approach
def tokenize_input(text: str) -> List[str]:
    # match words or non-word characters (e.g., punctuation)
    return re.findall(r'\w+|[^\w\s]', text)

# get the word frequencies
def get_word_frequencies(text: str) -> Dict[str, int]:
    words = tokenize_input(text)  
    return collections.Counter(words)

# bpe tokenizer implementation
class BPETokenizer:

    def __init__(self, vocab_size: int = 100):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = []
        self.end_of_word_marker = "</w>"
        
    def get_stats(self, vocab: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        # get frequency of adjacent symbol pairs
        pairs = collections.defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs
    
    def merge_vocab(self, pair: Tuple[str, str], vocab_in: Dict[str, int]) -> Dict[str, int]:
        # merge all instances of the most frequent pair
        vocab_out = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        
        for word in vocab_in:
            new_word = p.sub(''.join(pair), word)
            vocab_out[new_word] = vocab_in[word]
        return vocab_out
    
    def bpe_merge_once(self, vocab: Dict[str, int]) -> Tuple[Dict[str, int], Tuple[str, str], int]:
        # perform a single BPE merge step
        pairs = self.get_stats(vocab)
        if not pairs:
            return vocab, None, 0
            
        # find most frequent pair
        best_pair = max(pairs, key=pairs.get)
        pair_frequency = pairs[best_pair]
        new_vocab = self.merge_vocab(best_pair, vocab)
        
        return new_vocab, best_pair, pair_frequency
    
    def train(self, text: str) -> None:
        # train BPE tokenizer on the given text
        word_freqs = get_word_frequencies(text)
        
        # add end-of-word marker to each word
        vocab = {}
        for word, freq in word_freqs.items():
            # split word into characters and add end marker
            word_chars = ' '.join(list(word)) + ' ' + self.end_of_word_marker
            vocab[word_chars] = freq
        
        all_symbols = set()
        for word in vocab:
            all_symbols.update(word.split())
        
        self.vocab = {symbol: i for i, symbol in enumerate(sorted(all_symbols))}
        
        print("BPE Training")
        print(f"First 10 merges:")
        
        # perform merges until we reach desired vocabulary size
        for i in range(self.vocab_size - len(self.vocab)):
            vocab, pair, pair_freq = self.bpe_merge_once(vocab)
            if pair is None:
                break
                
            # store the merge
            self.merges.append(pair)
            
            # add merged token to vocabulary
            new_token = ''.join(pair)
            self.vocab[new_token] = len(self.vocab)
            
            # print first 10 merges with frequencies
            if i < 10:
                print(f"  Merge {i+1}: '{pair[0]}' + '{pair[1]}' -> '{new_token}' (frequency: {pair_freq})")
    
    def tokenize(self, text: str) -> List[str]:
        word_freqs = get_word_frequencies(text)
        
        vocab = {}
        for word, freq in word_freqs.items():
            word_chars = ' '.join(list(word)) + ' ' + self.end_of_word_marker
            vocab[word_chars] = freq
        
        for pair in self.merges:
            vocab = self.merge_vocab(pair, vocab)
        
        tokens = []
        for word, freq in vocab.items():
            word_tokens = word.split()
            tokens.extend(word_tokens * freq)
        
        return tokens

# WordPiece tokenizer implementation
class WordPieceTokenizer:
    
    def __init__(self, vocab_size: int = 100):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = []
        self.continuation_prefix = "##"
        
    def get_pair_scores(self, vocab: Dict[str, int]) -> Dict[Tuple[str, str], float]:
        pairs = collections.defaultdict(int)
        symbol_counts = collections.defaultdict(int)
        
        # count adjacent pairs and individual symbols
        for word, freq in vocab.items():
            symbols = word.split()
            # count individual symbols
            for symbol in symbols:
                symbol_counts[symbol] += freq
            # count adjacent pairs
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        
        # calculating scores: count(x▷y) / (count(x) * count(y))
        scores = {}
        for pair, pair_count in pairs.items():
            x_count = symbol_counts[pair[0]]
            y_count = symbol_counts[pair[1]]
            
            if x_count > 0 and y_count > 0:
                scores[pair] = pair_count / (x_count * y_count)
        
        return scores
    
    def merge_vocab(self, pair: Tuple[str, str], vocab_in: Dict[str, int]) -> Dict[str, int]:
        vocab_out = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        
        # Correct WordPiece merging logic
        if pair[1].startswith(self.continuation_prefix):
            # Second token is continuation: remove its ## prefix
            merged_token = pair[0] + pair[1][len(self.continuation_prefix):]
        else:
            # Second token is not continuation: keep as is
            merged_token = ''.join(pair)
        
        for word in vocab_in:
            new_word = p.sub(merged_token, word)
            vocab_out[new_word] = vocab_in[word]
        return vocab_out
    
    def wordpiece_merge_once(self, vocab: Dict[str, int]) -> Tuple[Dict[str, int], Tuple[str, str], float, int]:
        scores = self.get_pair_scores(vocab)
        if not scores:
            return vocab, None, 0.0, 0
            
        # find highest scoring pair
        best_pair = max(scores, key=scores.get)
        best_score = scores[best_pair]
        
        # get pair frequency
        pair_freq = 0
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                if (symbols[i], symbols[i + 1]) == best_pair:
                    pair_freq += freq
        
        new_vocab = self.merge_vocab(best_pair, vocab)
        
        return new_vocab, best_pair, best_score, pair_freq
    
    def train(self, text: str) -> None:
        # train WordPiece tokenizer on the given text
        word_freqs = get_word_frequencies(text)
        
        vocab = {}
        for word, freq in word_freqs.items():
            chars = list(word)
            if len(chars) > 1:
                word_chars = chars[0] + ' ' + ' '.join(self.continuation_prefix + c for c in chars[1:])
            else:
                word_chars = chars[0]
            vocab[word_chars] = freq
        
        all_symbols = set()
        for word in vocab:
            all_symbols.update(word.split())
        
        self.vocab = {symbol: i for i, symbol in enumerate(sorted(all_symbols))}
        
        print("WordPiece Training")
        print(f"First 10 merges:")
        
        for i in range(self.vocab_size - len(self.vocab)):
            vocab, pair, score, pair_freq = self.wordpiece_merge_once(vocab)
            if pair is None:
                break
                
            self.merges.append(pair)
            
            if pair[1].startswith(self.continuation_prefix):
                new_token = pair[0] + pair[1][len(self.continuation_prefix):]
            else:
                new_token = ''.join(pair)
            
            self.vocab[new_token] = len(self.vocab)
            
            # print first 10 merges with scores and frequencies
            if i < 10:
                print(f"  Merge {i+1}: '{pair[0]}' + '{pair[1]}' -> '{new_token}' (score: {score:.6f}, frequency: {pair_freq})")
    
    def tokenize(self, text: str) -> List[str]:
        # tokenize text using trained WordPiece model
        word_freqs = get_word_frequencies(text)
        
        # prepare vocabulary with continuation prefixes
        vocab = {}
        for word, freq in word_freqs.items():
            chars = list(word)
            if len(chars) > 1:
                word_chars = chars[0] + ' ' + ' '.join(self.continuation_prefix + c for c in chars[1:])
            else:
                word_chars = chars[0]
            vocab[word_chars] = freq
        
        # apply all learned merges in order
        for pair in self.merges:
            vocab = self.merge_vocab(pair, vocab)
        
        # extract all tokens
        tokens = []
        for word, freq in vocab.items():
            word_tokens = word.split()
            tokens.extend(word_tokens * freq)
        
        return tokens

# print the top-k most frequent tokens
def print_top_k(tokens: List[str], k: int = 10, title: str = "Top-K Tokens") -> None:
    # print the top-k most frequent tokens
    token_counts = collections.Counter(tokens)
    
    print(f"\n=== {title} ===")
    for i, (token, count) in enumerate(token_counts.most_common(k), 1):
        print(f"{i:2d}. '{token}': {count}")


def main():
    # file path for the corpus
    corpus_file = "q1_corpus_near1000.txt"
    
    print("BPE vs WordPiece Tokenizer Comparison")
    
    # read the corpus
    text = read_text_file(corpus_file)
    
    if not text:
        print("Failed to read corpus. Please ensure q1_corpus_near1000.txt is in the current directory.")
        return
    
    
    # Get word frequencies for training
    word_freqs = get_word_frequencies(text)
    
    # # Print word frequencies
    # print("Word Frequencies:")
    # for word, freq in sorted(word_freqs.items(), key=lambda x: x[1], reverse=True):
    #     print(f"  '{word}': {freq}")
    
    # train BPE tokenizer
    bpe_tokenizer = BPETokenizer(vocab_size=100)
    bpe_tokenizer.train(text)
    
    # show BPE final vocabulary with frequencies
    print(f"\nBPE Final Vocabulary ({len(bpe_tokenizer.vocab)} tokens) - Dictionary with frequencies:")
    bpe_tokens_sample = bpe_tokenizer.tokenize(text)
    bpe_vocab_freq = collections.Counter(bpe_tokens_sample)
    bpe_vocab_dict = {}
    for token in bpe_tokenizer.vocab.keys():
        bpe_vocab_dict[token] = bpe_vocab_freq[token]
    
    # display as dictionary format
    print("BPE_Vocab_Frequencies = {")
    for token, freq in sorted(bpe_vocab_dict.items(), key=lambda x: x[1], reverse=True):
        print(f"    '{token}': {freq},")
    print("}")
    
    # train WordPiece tokenizer  
    print("\n" + "=" * 50)
    wordpiece_tokenizer = WordPieceTokenizer(vocab_size=100)
    wordpiece_tokenizer.train(text)
    
    # show WordPiece final vocabulary with frequencies
    print(f"\nWordPiece Final Vocabulary ({len(wordpiece_tokenizer.vocab)} tokens) - Dictionary with frequencies:")
    wordpiece_tokens_sample = wordpiece_tokenizer.tokenize(text)
    wordpiece_vocab_freq = collections.Counter(wordpiece_tokens_sample)
    wordpiece_vocab_dict = {}
    for token in wordpiece_tokenizer.vocab.keys():
        wordpiece_vocab_dict[token] = wordpiece_vocab_freq[token]
    
    # display as dictionary format
    print("WordPiece_Vocab_Frequencies = {")
    for token, freq in sorted(wordpiece_vocab_dict.items(), key=lambda x: x[1], reverse=True):
        print(f"    '{token}': {freq},")
    print("}")
    
    # tokenize the original corpus and show top-10 frequencies
    print("\n\nTokenizing Original Corpus")
    
    bpe_tokens = bpe_tokenizer.tokenize(text)

    print_top_k(bpe_tokens, k=10, title="BPE Top-10 Tokens")
    
    wordpiece_tokens = wordpiece_tokenizer.tokenize(text)
    
    print_top_k(wordpiece_tokens, k=10, title="WordPiece Top-10 Tokens")


if __name__ == "__main__":
    main()
