import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv('emails.csv')

print(df.isna().sum())
df = df.fillna(0)

print(df)

df['BooleanSpam'] = (df['Category'] == 'spam').astype(int)

print(df)


message = CountVectorizer().fit_transform(df['Message'])


X_train, X_test, Y_train, Y_test = train_test_split(message, df['Category'], test_size=0.10, random_state=0)
print(message.shape)


classifier = MultinomialNB().fit(X_train, Y_train)
print(classifier.predict(X_train))
print(Y_train.values)

print(classifier.predict(X_test))
print(Y_test.values)


predict = classifier.predict(X_test)
print(classification_report(Y_test, predict))
print("\n")
print("ConfusionMatrix: \n", confusion_matrix(Y_test, predict))
print("\n")
print("Accuracy: \n", accuracy_score(Y_test, predict))