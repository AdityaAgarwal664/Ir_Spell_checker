import random
import nltk
from nltk.corpus import words
import pandas as pd

# Download the NLTK words corpus
nltk.download('words')
word_list = set(words.words())

# Function to generate a misspelled word
def generate_misspelled_word(word):
    # Randomly insert, delete, or replace a character
    action = random.choice(['insert', 'delete', 'replace'])
    word = list(word)  # Convert word to list for mutability
    
    if action == 'insert' and len(word) > 1:
        idx = random.randint(0, len(word)-1)
        word.insert(idx, random.choice('abcdefghijklmnopqrstuvwxyz'))
    elif action == 'delete' and len(word) > 1:
        idx = random.randint(0, len(word)-1)
        word.pop(idx)
    elif action == 'replace' and len(word) > 1:
        idx = random.randint(0, len(word)-1)
        word[idx] = random.choice('abcdefghijklmnopqrstuvwxyz')

    return ''.join(word)

# Generate 10,000 tuples
dataset = []
for _ in range(20000):
    word = random.choice(list(word_list))
    label = 1  # Correct word
    dataset.append((word, label))  # Correct word
    
    # Generate a misspelled word and append it to the dataset
    misspelled_word = generate_misspelled_word(word)
    label = 0  # Misspelled word
    dataset.append((misspelled_word, label))  # Misspelled word

# Convert to DataFrame
df = pd.DataFrame(dataset, columns=['word', 'label'])

# Save to CSV
df.to_csv('spell_check_dataset_20000.csv', index=False)

print("Dataset of 10,000 tuples generated and saved to 'spell_check_dataset_10000.csv'.")
