# %%
# Cell 1: Import necessary libraries
import os
import re
import numpy as np
from collections import defaultdict, Counter

# %%
# Cell 2: Function to clean and tokenize text
def clean_and_tokenize(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = text.split()  # Split by whitespace
    return tokens

# %%
# Cell 3: Function to generate N-grams from tokens
def generate_ngrams(tokens, n):
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    return ngrams

# %%
# Cell 4: Function to split data into training, validation, and test sets
def train_val_test_split(data, train_ratio=0.7, val_ratio=0.15, random_seed=42):
    np.random.seed(random_seed)
    np.random.shuffle(data)
    train_size = int(len(data) * train_ratio)
    val_size = int(len(data) * val_ratio)
    train_data = data[:train_size]
    val_data = data[train_size:train_size+val_size]
    test_data = data[train_size+val_size:]
    return train_data, val_data, test_data

# %%
# Cell 5: Naive Bayes Classifier class
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
                token_probability = (self.token_counts[label][token] + 1) / (self.class_totals[label] + len(self.vocabulary))
                log_probs[label] += np.log(token_probability)
        
        return max(log_probs, key=log_probs.get)

# %%
def load_data(folder_path, page_size, pages_per_book, ngram = 2):
    data = []
    labels = []
    filepaths = []
    
    # Get all .txt files and sort them
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    sorted_files = sorted(all_files)
    
    for book_num, filename in enumerate(sorted_files, 1):
        filepath = os.path.join(folder_path, filename)
        filepaths.append(filepath)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
            tokens = clean_and_tokenize(text)
            # Split the book into pages and limit to pages_per_book
            pages = [tokens[i:i+page_size] for i in range(0, len(tokens), page_size)][:pages_per_book]
            for page in pages:
                ngrams = generate_ngrams(page, n=ngram)  # Change n for different N-grams
                data.append(ngrams)
                labels.append(f'Book_{book_num}')
    
    return data, labels, filepaths

# %%
# Cell 7: Example usage
# Get all .txt files from the harry_potter_books folder
folder_path = 'harry_potter_books'
filepaths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.txt')]

print("Filepaths of the Harry Potter books:")
for filepath in filepaths:
    print(filepath)

# %%
# Use the function
folder_path = 'harry_potter_books'  # Update this to your actual folder path

#240,190S
data, labels, filepaths = load_data(folder_path, page_size=250, pages_per_book=240)
book_page_counts = Counter(labels)
# Print some information about the loaded data
print(f"\nTotal number of pages: {len(data)}")
print(f"Total number of books: {len(set(labels))}")

book_page_counts = Counter(labels)

print("\nNumber of pages per book:")
for book, count in sorted(book_page_counts.items()):
    print(f"{book}: {count} pages")

# Print a sample from the first book
print("\nSample from the first book:")
book1_sample = next(d for d, l in zip(data, labels) if l == 'Book_1')
print(f"First 10 tokens: {[token[0] for token in book1_sample[:10]]}")

# %%
# Count pages for each book
book_page_counts = Counter(labels)

# Sort the books by their number
sorted_books = sorted(book_page_counts.items(), key=lambda x: int(x[0].split('_')[1]))

print("Number of pages per book:")
for book, count in sorted_books:
    print(f"{book}: {count} pages")

# Calculate and print total pages
total_pages = sum(book_page_counts.values())
print(f"\nTotal pages across all books: {total_pages}")

# Calculate and print average pages per book
avg_pages = total_pages / len(book_page_counts)
print(f"Average pages per book: {avg_pages:.2f}")

# %%
combined_data = list(zip(data, labels))
train_data, val_data, test_data = train_val_test_split(combined_data)

print("Data split information:")
print(f"Train set size: {len(train_data)}")
print(f"Validation set size: {len(val_data)}")
print(f"Test set size: {len(test_data)}")

print("\nSample from train set (first 2 items):")
for i, (page_data, label) in enumerate(train_data[:2]):
    print(f"\nPage {i+1}:")
    print(f"Label: {label}")
    print(f"First 5 tokens: {[token[0] for token in page_data[:5]]}")

# %%
# Unpack the training data
train_pages, train_labels = zip(*train_data)

# Initialize and train the classifier
nb_classifier = NaiveBayesClassifier()
nb_classifier.train(train_pages, train_labels)

# %%
val_pages, val_labels = zip(*val_data)
correct = 0
total = len(val_pages)

for page, true_label in zip(val_pages, val_labels):
    predicted_label = nb_classifier.predict(page)
    if predicted_label == true_label:
        correct += 1

val_accuracy = correct / total
print(f"Validation Accuracy: {val_accuracy:.2%}")

# %%
test_pages, test_labels = zip(*test_data)
correct = 0
total = len(test_pages)

for page, true_label in zip(test_pages, test_labels):
    predicted_label = nb_classifier.predict(page)
    if predicted_label == true_label:
        correct += 1

test_accuracy = correct / total
print(f"Test Accuracy: {test_accuracy:.2%}")

# %%
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

y_true = test_labels
y_pred = [nb_classifier.predict(page) for page in test_pages]

cm = confusion_matrix(y_true, y_pred, labels=[f'Book_{i}' for i in range(1, 8)])

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=[f'Book {i}' for i in range(1, 8)],
            yticklabels=[f'Book {i}' for i in range(1, 8)])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# %%
import matplotlib.pyplot as plt

def count_pages_per_book(folder_path, page_size):
    book_page_counts = {}
    
    # Get all .txt files and sort them
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    sorted_files = sorted(all_files)
    
    for book_num, filename in enumerate(sorted_files, 1):
        filepath = os.path.join(folder_path, filename)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
            tokens = clean_and_tokenize(text)
            # Count the number of pages
            num_pages = len([tokens[i:i+page_size] for i in range(0, len(tokens), page_size)])
            book_page_counts[f'Book {book_num}'] = num_pages
    
    return book_page_counts

# Count pages using a specific page size
page_size = 1000  # You can adjust this value
book_page_counts = count_pages_per_book(folder_path, page_size)

# Create a bar plot
plt.figure(figsize=(12, 6))
books = list(book_page_counts.keys())
pages = list(book_page_counts.values())

plt.bar(books, pages)
plt.title(f'Number of Pages per Book (Page Size: {page_size} tokens)')
plt.xlabel('Book')
plt.ylabel('Number of Pages')

# Add value labels on top of each bar
for i, v in enumerate(pages):
    plt.text(i, v, str(v), ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Print the exact numbers
for book, count in book_page_counts.items():
    print(f"{book}: {count} pages")

# %%
# Run the naive bayes classifier with the amount of n grams set to 1, 2 ,3, 4 
# And plot the test accuracy for each of the n grams

n_values = [1, 2, 3, 4]
test_accuracies = []

for n in n_values:
    data, labels, filepaths = load_data(folder_path, page_size=250, pages_per_book=240, ngram=n)
    combined_data = list(zip(data, labels))
    train_data, val_data, test_data = train_val_test_split(combined_data)
    
    nb_classifier = NaiveBayesClassifier()
    nb_classifier.train(*zip(*train_data))
    
    test_pages, test_labels = zip(*test_data)
    correct = 0
    total = len(test_pages)
    
    for page, true_label in zip(test_pages, test_labels):
        predicted_label = nb_classifier.predict(page)
        if predicted_label == true_label:
            correct += 1
    
    test_accuracy = correct / total
    test_accuracies.append(test_accuracy)

print("Test Accuracies:")
for n, acc in zip(n_values, test_accuracies):
    print(f"N = {n}: {acc:.2%}")

# Plot a bar graph
plt.figure(figsize=(10, 6))
plt.bar(n_values, test_accuracies, color='skyblue')
plt.xlabel('N-gram Value')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy for Different N-gram Values')
plt.ylim(0, 1)
plt.xticks(n_values)
plt.tight_layout()
plt.show()




