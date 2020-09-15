from keras.models import Sequential
from keras import layers
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

from keras.utils import np_utils

from keras.models import Sequential
from keras import layers
from sklearn.feature_extraction.text import CountVectorizer

ammountCat ={}
def checkKey(dict, key):
    if key in dict.keys():
       ammountCat[dict[key]] = ammountCat[dict[key]] + 1
       return dict[key]
    else:
       dict[key] = len(dict) + 1
       ammountCat[dict[key]]=  1
       return dict[key]


import csv
f = open('purchase-order-data-2012-2015-.csv', 'r',encoding='UTF8')
reader = csv.reader(f)
headers = next(reader, None)
column = {}
columnIndex = {}
count = 0
for h in headers:
    column[h] = []
    columnIndex[h]=count
    count = count + 1
count = 0
limit = 50000
classDict = {}
columnNames = ['Item Name', 'Item Description', 'Class']
for row in reader:
   if row[ columnIndex['Class']] == "" or row[ columnIndex['Item Description']] == "" or row[ columnIndex['Item Name']] == "":
       continue
   column['Class'].append(checkKey(classDict, row[ columnIndex['Class']]))
   column['Item Description'].append(row[ columnIndex['Item Description']] +" " + row[ columnIndex['Item Name']])
   count = count + 1
   if count > 50000:
       break

cutoutThreshold = 500
for key, value in ammountCat.items():
    if value < cutoutThreshold:
        #print(key,":",value)
        found = True
        while found == True:
            broke = False
            for i in  range(len(column['Class'])):
               # print("i",column['Class'][i],key,i)
                if column['Class'][i] == key:
                    #print(key,column['Class'][i])
                    del column['Class'][i]
                    del column['Item Description'][i]
                    broke = True
                    break
            if broke == False:
                found = False
total =0
categories =0
for key, value in ammountCat.items():
    if value >= cutoutThreshold:
        categories = categories +1
        #print(key,":",value)
        total = value + total
print(len(column['Class']),len(column['Item Description']),total)
d = {'label': column['Class'], 'sentence': column['Item Description']}
column=""
df_yelp =pd.DataFrame(data=d)
sentences = df_yelp['sentence'].values
y = df_yelp['label'].values
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
y = np_utils.to_categorical(encoded_Y)
sentences_train, sentences_test, y_train, y_test = train_test_split(
   sentences, y, test_size=0.25, random_state=1000)


tokenizer = Tokenizer(num_words=5000)
print(sentences_train[0])
tokenizer.fit_on_texts(sentences_train)

X_train = tokenizer.texts_to_sequences(sentences_train)
X_test = tokenizer.texts_to_sequences(sentences_test)
print("---------------------------")
vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

print(sentences_train[2])
print(X_train[2])

maxlen = 100

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath,'r', encoding='UTF8') as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix
embedding_dim = 50
embedding_matrix = create_embedding_matrix(
    'glove.6B.50d.txt',
    tokenizer.word_index, embedding_dim)


model = Sequential()
model.add(layers.Embedding(vocab_size, embedding_dim,
                           weights=[embedding_matrix],
                           input_length=maxlen,
                           trainable=True))
model.add(layers.Conv1D(128, 5, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(categories*2, activation='relu'))
model.add(layers.Dense(categories, activation='softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train,
                    epochs=100,
                    verbose=False,
                    validation_data=(X_test, y_test),
                    batch_size=10)
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))