import tqdm
import numpy as np
import pandas as pd
import keras
import keras.metrics
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding, LSTM, Dropout, Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
import time




data = pd.read_csv("spam_or_not_spam.csv")

text = data["email"].astype(str).tolist()

# Text tokenization
# vectorizing text, turning each text into sequence of integers
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text)
# convert to sequence of integers
X = tokenizer.texts_to_sequences(text)

#Convert to arrays
X = np.array(X)
y = np.array(data["label"])

seq_len = max(len(i) for i in X)

X_new = pad_sequences(X, maxlen = 100)
y_new = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.25, random_state=7)

#####----- LOAD PRE-TRAINED EMBEDDING VECTORS FROM GLOVE -----#####

embedding_index = {}
with open("glove.6B.100d.txt", encoding='utf8') as f:
    for line in tqdm.tqdm(f, "Reading GloVe"):
            values = line.split()
            word = values[0]
            vectors = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = vectors
    
word_index = tokenizer.word_index
embedding_matrix = np.zeros((len(word_index)+1, 100))
for word, i in word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        # words not found will be 0s
        embedding_matrix[i] = embedding_vector

#####-----END LOAD -----#####
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def get_model(tokenizer, lstm_units):
    """
    Constructs the model,
    Embedding vectors => LSTM => 2 output Fully-Connected neurons with softmax activation
    """
    # get the GloVe embedding vectors
    
    model = Sequential()
    model.add(Embedding(len(tokenizer.word_index)+1,
              100,
              weights=[embedding_matrix],
              trainable=False,
              input_length=100))

    model.add(LSTM(lstm_units, recurrent_dropout=0.2))
    model.add(Dropout(0.3))
    model.add(Dense(2, activation="softmax"))
    # compile as rmsprop optimizer
    # aswell as with recall metric
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy",
                  metrics=['accuracy', precision_m, recall_m, f1_m])
    model.summary()
    return model

model = get_model(tokenizer=tokenizer, lstm_units=256)


# initialize our ModelCheckpoint and TensorBoard callbacks
# model checkpoint for saving best weights
model_checkpoint = ModelCheckpoint("results/spam_classifier_{val_loss:.2f}", save_best_only=True,
                                    verbose=1)
# for better visualization
tensorboard = TensorBoard("logs/spam_classifier_{time.time()}")
# print our data shapes
print("X_train.shape:", X_train.shape)
print("X_test.shape:", X_test.shape)
print("y_train.shape:", y_train.shape)
print("y_test.shape:", y_test.shape)
# train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test),
          batch_size=64, epochs=5,
          #callbacks=[tensorboard, model_checkpoint],
          verbose=1)

# get the loss and metrics
result = model.evaluate(X_test, y_test)
# extract those
loss = result[0]
accuracy = result[1]
precision = result[2]
recall = result[3]
f1 = result[4]

print(f"[+] Accuracy: {accuracy*100:.2f}%")
print(f"[+] Precision:   {precision*100:.2f}%")
print(f"[+] Recall:   {recall*100:.2f}%")
print(f"[+] F1:   {f1*100:.2f}%")

def get_predictions(text):
    sequence = tokenizer.texts_to_sequences([text])
    # pad the sequence
    sequence = pad_sequences(sequence, maxlen=100)
    # get the prediction
    prediction = model.predict(sequence)[0]
    return prediction

print(get_predictions('Arabian prince won 5000 dollars, you are the only winner congrats claim now!'))
