# Python 3 server example
from http.server import BaseHTTPRequestHandler, HTTPServer
from keras.models import Sequential
from keras import layers
import time
import math
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
import datetime
from numpy.random import seed
from sklearn.feature_extraction.text import CountVectorizer
#Declaring global variables
#diction used to pass around global values
dataset = "None"
#The cnn model
model = "None"
#the path to the CNN model
modelPath = "./models/cnn_model1"
#path to the SQLLite database
dbPath =  "dataset.db"
#struct used to keep track of how many entries below to a category
ammountCat ={}
#Dictionary for JSON parameter
parameters = {}
#database connection
conn = ""
#Utility function to create keep track of Target value
def checkKey(dict, key):
    if key in dict.keys():
       ammountCat[dict[key]] = ammountCat[dict[key]] + 1
       return key
    else:
       dict[key] = len(dict) + 1
       ammountCat[dict[key]]=  1
       return key
#Function to make word embbedding matrix
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
#load parameters from json file
def loadParameters(file):
    param = {}
    with open(file) as f:
        param = json.load(f)
    return param

#Utility function find the largest N elements out of an array
def Nmaxelements(list1, N):
    final_list = []
    for i in range(0, N):
        max1 = 0
        k = 0
        jVal = 0
        for j in range(len(list1)):
            if list1[j][1] > max1:
                max1 = list1[j][1];
                k = {"index":list1[j][0], "probability":list1[j][1] };
                jval = j
        del list1[jval];
        final_list.append(k)
    return final_list

#load data on startup
def loadDataSet():
    #connect to database
    conn = sqlite3.connect(dbPath)
    cur = conn.cursor()
    #select values used in current model
    cur.execute("SELECT * FROM dataset WHERE used_in_model = 1")
    rows = cur.fetchall()
    column = {}
    #Used to return a class description
    classDespDict = {}
    #used to map encoding to PSC code
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
    #Create data frame for Y and x values
    d = {'label': column['Class'], 'sentence': column['Item Description']}
    categories = len(classDespDict)
    df_yelp = pd.DataFrame(data=d)
    sentences = df_yelp['sentence'].values
    y = df_yelp['label'].values
    for i in range(len(y)):
        y[i] = str(y[i])
    #create encoder for predictions
    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_Y = encoder.transform(y)
    y = np_utils.to_categorical(encoded_Y)
    #Not actually used
    sentences_train, sentences_test, y_train, y_test = train_test_split(
        sentences, y, test_size=parameters["sampleParameters"]["testSize"], random_state=1000)
    #create tokenizer for predictions
    tokenizer = Tokenizer(num_words=parameters["sampleParameters"]["numWords"])
    tokenizer.fit_on_texts(sentences)
    return {"status": "LoadDataSet", "conn": cur, "tokenizer": tokenizer, "sentences_train": sentences_train,
            "sentences_test": sentences_test, "y_train": y_train, "y_test": y_test, "y": y,
            "classDescripDict": classDespDict, "encodingDict":encodingDict,'categorySize': categories,"weights":[], "db":conn}
#function to retrain model
def retrain(dataset):
    conn = dataset["conn"]
    # select distict classe
    conn.execute("SELECT DISTINCT class FROM dataset")
    rows = conn.fetchall()
    classes = []
    column = {}
    classDespDict = {}
    encodingDict = {}
    encodingDictRev= {}
    column["Class"] = []
    column["Item Description"] = []
    column["Class Description"] = []
    column["index_key"] = []
    column["weight"] = []
    classAmmount = {}
    totalNumber = 0
    #Find all the class counts
    for row in rows:
        classCountSql = "SELECT count(class) FROM dataset WHERE class = %s" % row[0]
        conn.execute(classCountSql)
        classCount = conn.fetchone()
        if classCount[0] > parameters['cutoutThreshold']:
            classes.append(row[0])
            classAmmount[row[0]]= classCount[0]
            totalNumber = totalNumber + classCount[0]
    # finc the amount of sample for a class to retrieve
    for key, value in classAmmount.items():
         classAmmount[key] = int(math.ceil((classAmmount[key]/totalNumber)*parameters["totalNumberOfSamples"]))
    #for class over threshold load
    for i in range(len(classes)):
        selectSql = "SELECT index_key, item_description, class , class_description , weight FROM dataset WHERE class = %s ORDER by weight DESC, timestamp DESC LIMIT %d"  % (classes[i], classAmmount[classes[i]])
        conn.execute(selectSql)
        rows = conn.fetchall()
        for row in rows:
            column["index_key"].append(row[0])
            column["Item Description"].append(row[1])
            column["Class"].append(row[2])
            column["Class Description"].append(row[3])
            column["weight"].append(row[4])
            classDespDict[row[2]] = row[3]
    categories = len(list(classAmmount.keys()))
    #create dataframe
    d = {'label': column['Class'], 'sentence': column['Item Description'],   'weight': column["weight"] }
    df_yelp = pd.DataFrame(data=d)
    sentences = df_yelp['sentence'].values
    weights = df_yelp['weight'].values
    y = df_yelp['label'].values
    for i in range(len(y)):
        y[i] = str(y[i])
    #create the encoder for y values
    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_Y = encoder.transform(y)
    eCount = 0
    classWeight = {}
    #assign class weights
    for en in  encoded_Y:
        encodingDict[en.item()] = y[eCount]
        encodingDictRev[y[eCount]] = en.item()
        eCount = eCount+1
        if en.item() in classWeight.keys():
            classWeight[en.item()] = classWeight[en.item()] + 1
        else:
            classWeight[en.item()] = 1
    #Assign low weight to outlies
    for key, value in classWeight.items():
        weight = value / parameters['totalNumberOfSamples']
        if weight < .0001:
            weight = .5
        else:
            weight = 1
        classWeight[key] = weight
    #Turn Y value into category so it can be used by model
    y = np_utils.to_categorical(encoded_Y)
    sentences_train, sentences_test, y_train, y_test,weights_train,weights_test = train_test_split(
        sentences, y,weights, test_size=parameters["sampleParameters"]["testSize"], random_state=1000)
    #Create tokenizers
    tokenizer = Tokenizer(num_words=parameters["sampleParameters"]["numWords"])
    tokenizer.fit_on_texts(sentences_train)
    conn.execute("UPDATE dataset SET used_in_model = 0, encoding_val = -999")
    for i in range(len(column['Class'])):
        sqlUpdate = "UPDATE dataset SET used_in_model = %d, encoding_val = %d WHERE index_key = %d" % (1,encodingDictRev[column['Class'][i]],column['index_key'][i])
        conn.execute(sqlUpdate)
    return {"status": "newDataSet", "conn": conn, "tokenizer": tokenizer,
            "sentences_train": sentences_train, "sentences_test": sentences_test, "y_train": y_train,
            "y_test": y_test, "y": y, "classDescripDict": classDespDict, "encodingDict": encodingDict,
            'categorySize': categories, "weights":weights_train, "db":dataset["db"],  "classWeight":classWeight}
#Create a initial models
def createNewDataSet():
    conn =""
    if path.exists(dbPath) == True:
        os.remove(dbPath)
    c = sqlite3.connect(dbPath)
    conn = c.cursor()
    # create table
    conn.execute('''CREATE TABLE  dataset ( index_key integer PRIMARY KEY autoincrement, item_description string, class string, 
    class_description string, encoding_val integer, weight integer, used_in_model integer,timestamp timestamp)''')
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
    trainingData = []
    count = 0
    # read in data from a CSV
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
    highest = 0
    #Find entries that need to be removed
    for key, value in ammountCat.items():
        if value > highest:
            highest = value
        if value > parameters['cutoutThreshold']:
            for i in range(len(column['Class'])):
                if classDict[column['Class'][i]] == key:
                   trainingData.append(i)
    total = 0
    categories = 0
    #find number of categories in the model
    for key, value in ammountCat.items():
        if value > parameters['cutoutThreshold']:
            categories = categories + 1
            total = value + total
    #print(len(column['Class']), len(column['Item Description']), total)
    d = {'label': column['Class'], 'sentence': column['Item Description']}
    df_yelp = pd.DataFrame(data=d)
    sentences = df_yelp['sentence'].values
    y = df_yelp['label'].values
    trainingSentence= []
    trainingY = []
    for i in range(len(trainingData)):
        trainingSentence.append(sentences[trainingData[i]])
        trainingY.append(y[trainingData[i]])
    sentences = np.asarray(trainingSentence)
    y = np.asarray(trainingY)
    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_Y = encoder.transform(y)
    eCount = 0
    classWeight = {}
    # map encoding data to categories (PSC)
    for en in  encoded_Y:
        encodingDict[en.item()] = y[eCount]
        encodingDictRev[y[eCount]] = en.item()
        eCount = eCount+1
        if  en.item() in classWeight.keys():
            classWeight[en.item()] = classWeight[en.item()] + 1
        else:
            classWeight[en.item()] =  1
    #Weight outlier classes
    for key, value in classWeight.items():
        weight = value/parameters['totalNumberOfSamples']
        if weight < .0001:
            weight = .5
        else:
            weight = 1
        classWeight[key] = weight
    y = np_utils.to_categorical(encoded_Y)
    #load data into the database
    for i in range(len(column['Class'])):
        encodingValue = -999
        if column['Class'][i] in encodingDictRev.keys():
            encodingValue =  encodingDictRev[column['Class'][i]]
        usedInModel = 0
        if i in trainingData:
            usedInModel= 1
        conn.execute('INSERT INTO dataset (item_description, class , class_description , encoding_val, weight, used_in_model,timestamp ) VALUES (?,?,?,?,?,?,?)',
                     [column['Item Description'][i], column['Class'][i], column['Class Description'][i], encodingValue, 1, usedInModel, datetime.datetime.now()])
    c.commit()
    print(len(trainingSentence), len(trainingY), total)
    sentences_train, sentences_test, y_train, y_test = train_test_split(
        sentences, y, test_size=parameters["sampleParameters"]["testSize"], random_state=1000)
    tokenizer = Tokenizer(num_words=parameters["sampleParameters"]["numWords"])
    tokenizer.fit_on_texts(sentences_train)

    return {"status":"newDataSet","conn":conn,"tokenizer": tokenizer,
            "sentences_train":sentences_train,"sentences_test":sentences_test,"y_train":y_train,
            "y_test":y_test, "y":y, "classDescripDict":classDespDict,"encodingDict":encodingDict,  'categorySize':categories,"weights":[],"db":c, "classWeight":classWeight}
#function to train CNN mode
def trainCNN(dataset):
    tokenizer = dataset['tokenizer']
    sentences_train = dataset['sentences_train']
    sentences_test= dataset['sentences_test']
    y_train = dataset['y_train']
    y_test=dataset['y_test']
    categorySize = dataset['categorySize']
    wts = dataset["weights"]
    classWeight = dataset['classWeight']
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
    # create sequential model
    model = Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim,weights=[embedding_matrix],input_length=maxlen,trainable=True))
    model.add(layers.Conv1D(parameters["cnnParameters"]["numberOfFeatures"], parameters["cnnParameters"]["kernalSize"], activation='relu'))
    model.add(layers.Conv1D(parameters["cnnParameters"]["numberOfFeatures"], parameters["cnnParameters"]["kernalSize"], activation='relu'))
    model.add(layers.Dropout(rate=.1))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(categorySize*2, activation='relu'))
    model.add(layers.Dense(categorySize, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    history = None
    #train initial model
    if len(wts) == 0:
        history = model.fit(X_train, y_train,
                        epochs=parameters["cnnParameters"]["epochs"],
                        verbose=True,
                        validation_data=(X_test, y_test),
                        class_weight = classWeight,
                        batch_size=parameters["cnnParameters"]["batchSize"])
    #retrain model
    else:
        print(len(wts), len(X_train), len(y_test))
        history = model.fit(X_train, y_train,
                        epochs=parameters["cnnParameters"]["epochs"],
                        verbose=True,
                        sample_weight=wts,
                        validation_data=(X_test, y_test),
                        class_weight = classWeight,
                        batch_size=parameters["cnnParameters"]["batchSize"])
    # convert the history.history dict to a pandas DataFrame:
    hist_df = pd.DataFrame(history.history)

    # save to json:
    #save stats
    hist_json_file = 'history.json'
    with open(hist_json_file, mode='w') as f:
        hist_df.to_json(f)

    # or save to csv:
    hist_csv_file = 'history.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)
    loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))
    model.save(modelPath)
    return {"model": model,"loss":loss, "accuracy": accuracy}
#load model on startup
def loadModel():
    return keras.models.load_model(modelPath)
#function to predrict list of PSC codes
def predict(txt, dataset, model):
    tokenizer = dataset['tokenizer']
    encodingDict = dataset['encodingDict']
    classDescripDict = dataset['classDescripDict']
    jsonRequest = json.loads(txt)
    test_sent = [jsonRequest["text"]]
    maxlen = parameters["cnnParameters"]["maxlen"]
    test = tokenizer.texts_to_sequences(test_sent)
    test = pad_sequences(test, padding='post', maxlen=maxlen)
    classPredVals = model.predict(test)
    classPredVals = classPredVals[0]
    listOfClass = []
    for i in range(len(classPredVals)):
        print(classPredVals[i])
        listOfClass.append([i,classPredVals[i]])
    numOfPrediction = 5
    if len(classPredVals) <5:
        numOfPrediction = len(classPredVals)
    predictions = Nmaxelements(listOfClass,numOfPrediction)
    rsps = {}
    rsps["predictions"] = []
    for i in range(len(predictions)):
        rsp = {}
        rsp["PSC"] = str(encodingDict[predictions[i]["index"]])
        rsp["desc"] = classDescripDict[encodingDict[predictions[i]["index"]]]
        rsp["probability"] = str(predictions[i]["probability"])
        rsp["status"] = "success"
        rsps["predictions"].append(rsp)
    print(rsps)
    y = json.dumps(rsps)
    return y
#Function to save correction and/or predictions
def accept(txt, dataset):
    classDescripDict = dataset['classDescripDict']
    cur = dataset['conn']
    conn = dataset['db']
    jsonRequest = json.loads(txt)
    descrip =  "UNKNOWN"
    if  jsonRequest["PSC"]  in classDescripDict.keys():
        descrip = classDescripDict[jsonRequest["PSC"]]

    if jsonRequest["save"] == True:
        cur.execute('INSERT INTO dataset (item_description, class , class_description , encoding_val, weight, used_in_model, timestamp )  VALUES (?,?,?,?,?,?,?)',
                    [jsonRequest["text"], jsonRequest["PSC"] , descrip, -999, 1.5, 0, datetime.datetime.now()])
        conn.commit()
    rsp = {}
    rsp["status"] = "success"
    y = json.dumps(rsp)
    return y


#Load json parameters
parameters = loadParameters('parameters.json')
import time
hostName = "localhost"
serverPort = 8080
model = None
dataset = None
#HTTP Server code
class MyServer(BaseHTTPRequestHandler):
    #Code to handle POST request
    def do_POST(self):
        global dataset
        global model
        content_len = int(self.headers.get('Content-Length',0))
        post_body = self.rfile.read(content_len).decode('UTF-8')
        print(post_body)
        resBody = ""
        resBody = '{"status":"error"}'
        #Prediction endpoint
        if self.path == "/predict":
            resBody = predict(post_body,dataset,model)
        #accept endpoint
        if self.path == "/accept":
            resBody = accept(post_body,dataset)
        #retrain endpoint
        if  self.path == "/retrain":
            dataset = retrain(dataset)
            start_time = time.time()
            trainCNN(dataset)
            print("--- %s seconds ---" % (time.time() - start_time))
            model = loadModel()
            rsp ={}
            rsp["status"] = "success"
            print(rsp)
            resBody = json.dumps(rsp)
            dataset["db"].commit()
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(bytes(resBody, "utf-8"))
# server start coode
#if __name__ == "__main__":
def startServer():
    global dataset
    global model
    webServer = HTTPServer((hostName, serverPort), MyServer)
    #See if initial model has already been created
    if path.exists(dbPath) == False or path.exists(modelPath) == False:
        dataset = createNewDataSet()
        start_time = time.time()
        trainCNN(dataset)
        print("--- %s seconds ---" % (time.time() - start_time))
        model = loadModel()
    else:
        #load data
        dataset = loadDataSet()
        model = loadModel()

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")

startServer()
