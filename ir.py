import nltk
nltk.download('words')
from nltk.corpus import words
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


#nltk.download('words')
word_list = set(words.words()) 


def is_misspelled(word):
    return word.lower() not in word_list

def edit_distance(str1, str2):
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
                dp[i][j] = min(dp[i - 1][j],      
                               dp[i][j - 1],       
                               dp[i - 1][j - 1]) + 1  
    
    return dp[m][n]

def suggest_corrections(word, n=3):
    suggestions = [(w, edit_distance(word, w)) for w in word_list]
    suggestions.sort(key=lambda x: x[1])
    return [s[0] for s in suggestions[:n]]


df = pd.read_csv('spell_check_dataset_10000.csv')


vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 3))
X = vectorizer.fit_transform(df['word'])
y = df['label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression().fit(X_train, y_train)

def spell_checker(word):
    prediction = model.predict(vectorizer.transform([word]))[0]
    if prediction == 1:
        return f"{word} is spelled correctly."
    else:
        corrections = suggest_corrections(word)
        best_correction = corrections[0]
        distance = edit_distance(word, best_correction)
        return f"Suggestions: {', '.join(corrections)} | Closest correction: {best_correction} (Edit distance: {distance})"


ans=input("Enter the word: ")
print(spell_checker(ans))
