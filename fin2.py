import tqdm
import numpy as np
import pandas as pd
import keras
from keras.metrics import Precision, Recall
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding, LSTM, Dense, Bidirectional
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split




data = pd.read_csv("spam_or_not_spam.csv")

x = data["email"].astype(str).tolist()
y = data["label"]

#y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
# Text tokenization
# vectorizing text, turning each text into sequence of integers
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
# convert to sequence of integers
seq = tokenizer.texts_to_sequences(X_train)


pad_seq = pad_sequences(seq, maxlen = 200)


vocab_size = len(tokenizer.word_index) + 1

#####----- LOAD PRE-TRAINED EMBEDDING VECTORS FROM GLOVE -----#####

embedding_index = {}
with open("glove.6B.200d.txt", encoding='utf8') as f:
    for line in tqdm.tqdm(f, "Reading GloVe"):
            values = line.split(' ')
            word = values[0]
            vectors = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = vectors


embedding_matrix = np.zeros((vocab_size, 200))
for word, i in tqdm.tqdm(tokenizer.word_index.items()):
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        # words not found will be 0
        embedding_matrix[i] = embedding_vector

#####--------------          END LOAD        --------------#####

model = Sequential()
model.add(Embedding(vocab_size, 200, weights = [embedding_matrix], input_length=200, trainable = False))
model.add(Bidirectional(LSTM(75)))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy', Precision(), Recall()])
history = model.fit(pad_seq, y_train, epochs = 10, batch_size=256, validation_split=0.25)

X_test = tokenizer.texts_to_sequences(X_test)
testing_seq = pad_sequences(X_test,maxlen=200)


result = model.evaluate(testing_seq, y_test)

predict = model.predict_classes(testing_seq)

loss = result[0]
accuracy = result[1]
precision = result[2]
recall = result[3]

print('\n')
print(f"[+] Accuracy: {accuracy*100:.2f}%")
print(f"[+] Precision:   {precision*100:.2f}%")
print(f"[+] Recall:   {recall*100:.2f}%")
print(f"[+] F1 Score:   {(2*(recall*precision)/(recall+precision))*100:.2f}%")


def get_predictions(text):
    sequence = tokenizer.texts_to_sequences([text])
    # pad the sequence
    sequence = pad_sequences(sequence, maxlen=200)
    # get the prediction
    prediction = model.predict(sequence)
    return prediction

print(get_predictions('Arabian prince won 5000 dollars, you are the only winner congrats claim now spam mail spam mail!'))
print(get_predictions('john remember to bring all the necessary files tomorrow at work please. It is very important and vital and crusial fuck you john we are breaking up'))
