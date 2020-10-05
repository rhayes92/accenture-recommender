from keras.models import Sequential
from keras import layers
import time
import keras
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import os
import sqlite3
from os import path
from keras.utils import np_utils
import csv
from keras.models import Sequential
from keras import layers
from sklearn.feature_extraction.text import CountVectorizer

ammountCat ={}
parameters = {}
conn = ""
def checkKey(dict, key):
    if key in dict.keys():
       ammountCat[dict[key]] = ammountCat[dict[key]] + 1
       return key
    else:
       dict[key] = len(dict) + 1
       ammountCat[dict[key]]=  1
       return key

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
def loadParameters(file):
    param = {}
    with open(file) as f:
        param = json.load(f)
    return param
def loadDataSet():
    conn = sqlite3.connect('dataset.db')
    cur = conn.cursor()
    cur.execute("SELECT * FROM dataset")
    rows = cur.fetchall()
    column = {}
    classDespDict = {}
    encodingDict = {}
    column["Class"] = []
    column["Item Description"] = []
    column["Class Description"] = []
    for row in rows:
        column['Item Description'].append(row[1])
        column['Class'].append(row[2])
        column['Class Description'].append(row[3])
        encodingDict[row[4]] = row[2]
        classDespDict[row[2]] = row[3]
    d = {'label': column['Class'], 'sentence': column['Item Description']}
    categories = len(classDespDict)
    df_yelp = pd.DataFrame(data=d)
    sentences = df_yelp['sentence'].values
    y = df_yelp['label'].values
    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_Y = encoder.transform(y)
    y = np_utils.to_categorical(encoded_Y)
    sentences_train, sentences_test, y_train, y_test = train_test_split(
        sentences, y, test_size=parameters["sampleParameters"]["testSize"], random_state=1000)
    tokenizer = Tokenizer(num_words=parameters["sampleParameters"]["numWords"])
    tokenizer.fit_on_texts(sentences_train)
    return {"status": "LoadDataSet", "conn": cur, "tokenizer": tokenizer, "sentences_train": sentences_train,
            "sentences_test": sentences_test, "y_train": y_train, "y_test": y_test, "y": y,
            "classDescripDict": classDespDict, "encodingDict":encodingDict,'categorySize': categories}

def createNewDataSet():
    conn =""
    if path.exists('dataset.db') == True:
        os.remove("dataset.db")
    c = sqlite3.connect('dataset.db')
    conn = c.cursor()
    conn.execute('''CREATE TABLE  dataset ( index_key integer PRIMARY KEY, item_description string, class string, class_description string, encoding_val integer)''')
    conn.execute('''CREATE TABLE  entry (entry_key integer PRIMARY KEY, new_entries integer) ''')
    f = open(parameters['sampleFile'], 'r', encoding='UTF8')
    reader = csv.reader(f)
    headers = next(reader, None)
    column = {}
    columnIndex = {}
    classDict = {}
    classDespDict = {}
    encodingDict = {}
    encodingDictRev= {}
    count = 0
    for h in headers:
        columnIndex[h] = count
        count = count + 1
    column["Class"] = []
    column["Item Description"] = []
    column["Class Description"] = []
    count = 0
    for row in reader:
        if row[columnIndex[parameters["yVal"]]] == "":
            continue
        blankVal = False
        itemDescrip= ''
        for i in  parameters['xVals']:
            if row[columnIndex[i]] == "":
                blankVal  =  True
                break
            itemDescrip = itemDescrip +' '+ row[columnIndex[i]]
        if blankVal == True:
            continue
        column['Class'].append(checkKey(classDict, row[columnIndex[parameters['yVal']]]))
        column['Item Description'].append(itemDescrip)
        column['Class Description'].append(row[columnIndex[parameters["yDescription"]]])
        classDespDict[row[columnIndex[parameters['yVal']]]] = row[columnIndex[parameters["yDescription"]]]
        count = count + 1
        if count > parameters['totalNumberOfSamples']:
            break
    for key, value in ammountCat.items():
        if value < parameters['cutoutThreshold']:
            print(key, ":", value)
            found = True
            while found == True:
                broke = False
                for i in range(len(column['Class'])):
                    if classDict[column['Class'][i]] == key:
                        # print(key,column['Class'][i])
                        del column['Class'][i]
                        del column['Item Description'][i]
                        del column['Class Description'][i]
                        broke = True
                        break
                if broke == False:
                    found = False
    total = 0
    categories = 0
    for key, value in ammountCat.items():
        if value >= parameters['cutoutThreshold']:
            categories = categories + 1
            total = value + total
    print(len(column['Class']), len(column['Item Description']), total)
    d = {'label': column['Class'], 'sentence': column['Item Description']}
    df_yelp = pd.DataFrame(data=d)
    sentences = df_yelp['sentence'].values
    y = df_yelp['label'].values
    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_Y = encoder.transform(y)
    eCount = 0
    for en in  encoded_Y:
        encodingDict[en.item()] = y[eCount]
        encodingDictRev[y[eCount]] = en.item()
        eCount = eCount+1
    y = np_utils.to_categorical(encoded_Y)
    for i in range(len(column['Class'])):
        #print(type(encodingDictRev[column['Class'][i]]))
        conn.execute('INSERT INTO dataset VALUES (?,?,?,?,?)',
                     [i, column['Item Description'][i], column['Class'][i], column['Class Description'][i], encodingDictRev[column['Class'][i]]])
    c.commit()
    sentences_train, sentences_test, y_train, y_test = train_test_split(
        sentences, y, test_size=parameters["sampleParameters"]["testSize"], random_state=1000)
    tokenizer = Tokenizer(num_words=parameters["sampleParameters"]["numWords"])
    tokenizer.fit_on_texts(sentences_train)
    return {"status":"newDataSet","conn":conn,"tokenizer": tokenizer,
            "sentences_train":sentences_train,"sentences_test":sentences_test,"y_train":y_train,
            "y_test":y_test, "y":y, "classDescripDict":classDespDict,"encodingDict":encodingDict,  'categorySize':categories}

def trainCNN(dataset):
    tokenizer = dataset['tokenizer']
    sentences_train = dataset['sentences_train']
    sentences_test= dataset['sentences_test']
    y_train = dataset['y_train']
    y_test=dataset['y_test']
    categorySize = dataset['categorySize']
    X_train = tokenizer.texts_to_sequences(sentences_train)
    X_test = tokenizer.texts_to_sequences(sentences_test)
    vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
    maxlen =  parameters["cnnParameters"]["maxlen"]
    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
    embedding_dim = 50
    embedding_matrix = create_embedding_matrix(
        parameters["cnnParameters"]["maxtrixEmbeddingFile"],
        tokenizer.word_index, embedding_dim)

    model = Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim,
                               weights=[embedding_matrix],
                               input_length=maxlen,
                               trainable=True))
    model.add(layers.Conv1D(parameters["cnnParameters"]["numberOfFeatures"], parameters["cnnParameters"]["kernalSize"], activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(categorySize*2, activation='relu'))
    model.add(layers.Dense(categorySize, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    history = model.fit(X_train, y_train,
                        epochs=parameters["cnnParameters"]["epochs"],
                        verbose=False,
                        validation_data=(X_test, y_test),
                        batch_size=parameters["cnnParameters"]["batchSize"])
    loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))
    model.save("./models/cnn_model")
    return {"model": model,"loss":loss, "accuracy": accuracy}

def loadModel():
    return keras.models.load_model("./models/cnn_model")


def predict(dataset, model):
    tokenizer = dataset['tokenizer']
    encodingDict = dataset['encodingDict']
    classDescripDict = dataset['classDescripDict']
    test_sent = ["Apple Soda"]
    maxlen = parameters["cnnParameters"]["maxlen"]
    test = tokenizer.texts_to_sequences(test_sent)
    test = pad_sequences(test, padding='post', maxlen=maxlen)
    ynew = model.predict(test)
    ynew2 = model.predict_classes(test)
    ynew3 = model.predict_proba(test)
    for i in range(len(test)):
       # print("X=%s, Predicted=%s" % (test[i], ynew[i]))
        classV = ynew2[i].item()
        print (classV)
        print(encodingDict[classV], classDescripDict[encodingDict[classV]])
        #print("X=%s, Predicted3=%s" % (test[i], ynew3[i]))

parameters = loadParameters('parameters.json')
dataset = createNewDataSet()
start_time = time.time()
trainCNN(dataset)
print("--- %s seconds ---" % (time.time() - start_time))
#dataset = loadDataSet()
model = loadModel()
predict(dataset,model)