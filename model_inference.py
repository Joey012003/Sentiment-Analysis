import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv("Emotion_Dataset.csv")
print(df.head())

sns.countplot(x='emotion', data=df)
plt.title("Emotion Counts")
plt.show()

import nltk
nltk.download('stopwords')

import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text.lower())
    words = [ps.stem(w) for w in text.split() if w not in stop_words]
    return ' '.join(words)
df['cleaned'] = df['sentence'].apply(clean_text)

cv = CountVectorizer()
X = cv.fit_transform(df['cleaned'])
y = df['emotion']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

for i in range(5):
    print("Text:", df['sentence'].iloc[i]) # Changed 'text' to 'sentence'
    print("Predicted Emotion:", model.predict(cv.transform([df['cleaned'].iloc[i]]))[0])