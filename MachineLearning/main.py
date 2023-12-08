import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings('ignore')
sns.set()

# Load the dataset (Make sure the path is correct)
imdb = pd.read_csv("IMDB Dataset.csv")
print(imdb.head())

# Display value counts for sentiment
print(imdb.sentiment.value_counts())

# Example text processing
text = imdb['review'][0]
print(text)
print("<================>")
print(word_tokenize(text))

# Corpus creation
corpus = []
for text in imdb['review']:
    words = [word.lower() for word in word_tokenize(text)]
    corpus.append(words)

# Calculate the number of words in the corpus
num_words = len(corpus)
print(num_words)

# Dataset shape
print(imdb.shape)

# Train-test split
train_size = int(imdb.shape[0] * 0.8)
X_train = imdb.review[:train_size]
Y_train = imdb.sentiment[:train_size]

X_test = imdb.review[train_size:]
Y_test = imdb.sentiment[train_size:]

# Text tokenization and padding
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_train = pad_sequences(X_train, maxlen=128, truncating='post', padding='post')

X_test = tokenizer.texts_to_sequences(X_test)
X_test = pad_sequences(X_test, maxlen=128, truncating='post', padding='post')

# Label encoding
le = LabelEncoder()
Y_train = le.fit_transform(Y_train)
Y_test = le.transform(Y_test)

# Display shapes
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

# Model creation
model = Sequential()
model.add(Embedding(input_dim=num_words, output_dim=100, input_length=128, trainable=True))
model.add(LSTM(100, dropout=0.1, return_sequences=True))
model.add(LSTM(100, dropout=0.1))
model.add(Dense(1, activation='sigmoid'))

# Model compilation
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Model training
history = model.fit(X_train, Y_train, epochs=5, batch_size=64, validation_data=(X_test, Y_test))


plt.figure(figsize=(16, 5))
epochs = range(1, len(history.history['accuracy']) + 1)
plt.plot(epochs, history.history['loss'], 'b', label='Training Loss', color='red')
plt.plot(epochs, history.history['val_loss'], 'b', label='Validation Loss')
plt.legend()
plt.show()

plt.figure(figsize=(16, 5))
epochs = range(1, len(history.history['accuracy']) + 1)
plt.plot(epochs, history.history['accuracy'], 'b', label='Training Accuracy', color='red')
plt.plot(epochs, history.history['val_accuracy'], 'b', label='Validation Accuracy')
plt.legend()
plt.show()

validation_sentence = ["It had some good parts like the acting was pretty good but the story was not impressing at all."]
validation_sentence_tokenized = tokenizer.texts_to_sequences(validation_sentence)
validation_sentence_padded = pad_sequences(validation_sentence_tokenized, maxlen=128, truncating='post', padding='post')

print(validation_sentence_padded[0])
print("Probability of Positive: {}".format(model.predict(validation_sentence_padded)[0]))


validation_sentence = ["It had some mediocre parts like the storyline although the actors performed really well and that is why overall I enjoyed it."]
validation_sentence_tokenized = tokenizer.texts_to_sequences(validation_sentence)
validation_sentence_padded = pad_sequences(validation_sentence_tokenized, maxlen=128, truncating='post', padding='post')

print(validation_sentence_padded[0])
print("Probability of Positive: {}".format(model.predict(validation_sentence_padded)[0]))

validation_sentence = ["this movie was bad."]
validation_sentence_tokenized = tokenizer.texts_to_sequences(validation_sentence)
validation_sentence_padded = pad_sequences(validation_sentence_tokenized, maxlen=128, truncating='post', padding='post')

print(validation_sentence_padded[0])
print("Probability of Positive: {}".format(model.predict(validation_sentence_padded)[0]))