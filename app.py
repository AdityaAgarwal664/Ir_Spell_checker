from flask import Flask, render_template, request, jsonify
import nltk
from nltk.corpus import words
from nltk.metrics.distance import edit_distance
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

app = Flask(__name__)

# Download NLTK words corpus
nltk.download('words')
word_list = set(words.words())

# Load the dataset for training
df = pd.read_csv('spell_check_dataset_10000.csv')

# Vectorizer and model setup for logistic regression
vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 3))
X = vectorizer.fit_transform(df['word'])
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression().fit(X_train, y_train)

# Function to check if a word is in the NLTK words list
def is_misspelled(word):
    return word.lower() not in word_list

# Function to calculate the edit distance between two words
def edit_distance_func(str1, str2):
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
    
    return dp[m][n]

# Function to suggest corrections based on the edit distance
def suggest_corrections(word, n=3):
    suggestions = [(w, edit_distance_func(word, w)) for w in word_list]
    suggestions.sort(key=lambda x: x[1])
    return [s[0] for s in suggestions[:n]]

# Function to check the spelling of a word using the trained model
def spell_checker(word):
    # First check if the word is spelled correctly using the trained model
    prediction = model.predict(vectorizer.transform([word]))[0]
    
    if prediction == 1:
        return f"{word} is spelled correctly."
    else:
        # If not, suggest corrections based on the edit distance
        corrections = suggest_corrections(word)
        best_correction = corrections[0]
        distance = edit_distance_func(word, best_correction)
        return {
            'word': word,
            'suggestions': corrections,
            'closest_correction': best_correction,
            'edit_distance': distance
        }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/check', methods=['POST'])
def check_spellings():
    # Get the text from the form
    text = request.form['text']
    words_in_text = text.split()  # Split text into words
    results = []
    
    for word in words_in_text:
        result = spell_checker(word)
        if "is spelled correctly" not in result:
            results.append(result)
    
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
