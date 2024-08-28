import re
import numpy as np
from collections import defaultdict, Counter

# Function to clean and tokenize text


def clean_and_tokenize(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = text.split()  # Split by whitespace
    return tokens

# Function to generate N-grams from tokens


def generate_ngrams(tokens, n):
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    return ngrams

# Function to split data into training, validation, and test sets


def train_val_test_split(data, train_ratio=0.7, val_ratio=0.15):
    np.random.shuffle(data)
    train_size = int(len(data) * train_ratio)
    val_size = int(len(data) * val_ratio)
    train_data = data[:train_size]
    val_data = data[train_size:train_size+val_size]
    test_data = data[train_size+val_size:]
    return train_data, val_data, test_data

# Naive Bayes Classifier class


class NaiveBayesClassifier:
    def __init__(self):
        self.class_counts = defaultdict(int)
        self.token_counts = defaultdict(lambda: defaultdict(int))
        self.class_totals = defaultdict(int)
        self.vocabulary = set()

    def train(self, data, labels):
        for tokens, label in zip(data, labels):
            self.class_counts[label] += 1
            self.class_totals[label] += len(tokens)
            for token in tokens:
                self.token_counts[label][token] += 1
                self.vocabulary.add(token)

    def predict(self, tokens):
        log_probs = {}
        total_docs = sum(self.class_counts.values())

        for label in self.class_counts:
            log_probs[label] = np.log(self.class_counts[label] / total_docs)
            for token in tokens:
                token_probability = (
                    self.token_counts[label][token] + 1) / (self.class_totals[label] + len(self.vocabulary))
                log_probs[label] += np.log(token_probability)

        return max(log_probs, key=log_probs.get)

# Load and preprocess the text data


def load_data(filepaths):
    data = []
    labels = []

    for i, filepath in enumerate(filepaths):
        with open(filepath, 'r') as f:
            text = f.read()
            tokens = clean_and_tokenize(text)
            # Change n for different N-grams
            ngrams = generate_ngrams(tokens, n=1)
            data.append(ngrams)
            labels.append(f'Book_{i+1}')

    return data, labels


# Example usage
if __name__ == '__main__':
    # Filepaths for the Harry Potter books (replace with actual paths)
    filepaths = ['book1.txt', 'book2.txt', 'book3.txt']

    # Load data
    data, labels = load_data(filepaths)

    # Split data into train, validation, and test sets
    train_data, val_data, test_data = train_val_test_split(
        list(zip(data, labels)))

    # Separate data and labels
    train_tokens, train_labels = zip(*train_data)
    val_tokens, val_labels = zip(*val_data)
    test_tokens, test_labels = zip(*test_data)

    # Initialize and train the Naive Bayes Classifier
    nbc = NaiveBayesClassifier()
    nbc.train(train_tokens, train_labels)

    # Evaluate on validation set
    correct = 0
    for tokens, label in zip(val_tokens, val_labels):
        prediction = nbc.predict(tokens)
        if prediction == label:
            correct += 1
    accuracy = correct / len(val_labels)
    print(f'Validation Accuracy: {accuracy * 100:.2f}%')

    # Evaluate on test set (repeat similar steps for the test set)
