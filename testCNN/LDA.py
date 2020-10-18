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
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)
import nltk
nltk.download('wordnet')

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
limit = 1000000
classDict = {}
columnNames = ['Item Name', 'Item Description', 'Class']
for row in reader:
   if row[ columnIndex['Class']] == "" or row[ columnIndex['Item Description']] == "" or row[ columnIndex['Item Name']] == "":
       continue
   column['Class'].append(checkKey(classDict, row[ columnIndex['Class']]))
   column['Item Description'].append(row[ columnIndex['Item Description']] +" " + row[ columnIndex['Item Name']])
   count = count + 1
   if count > limit:
       break

cutoutThreshold = 10
for key, value in ammountCat.items():
    if value < cutoutThreshold:
        print(key,":",value)
        found = True
        while found == True:
            broke = False
            for i in  range(len(column['Class'])):
                #print("i:",column['Class'][i],key,i)
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
        print("-",key,":",value)
        total = value + total
print(len(column['Class']),len(column['Item Description']),total)
d = {'label': column['Class'], 'sentence': column['Item Description']}
column=""
df_yelp =pd.DataFrame(data=d)
data_text = df_yelp[['sentence']]
data_text['index'] = df_yelp.index
documents = data_text

stemmer = SnowballStemmer("english")
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result
print(documents)
doc_sample = documents[documents['index'] == 39].values[0][0]
print('original document: ')
words = []
for word in doc_sample.split(' '):
    words.append(word)
print(words)
print('\n\n tokenized and lemmatized document: ')
print(preprocess(doc_sample))
processed_docs = documents['sentence'].map(preprocess)
print(processed_docs[:10])
dictionary = gensim.corpora.Dictionary(processed_docs)
count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break
dictionary.filter_extremes(no_below=20, no_above=0.5, keep_n=1000000)
print(len(dictionary),len(documents))
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
bow_doc_4310 = bow_corpus[90]
for i in range(len(bow_doc_4310)):
    print("Word {} (\"{}\") appears {} time.".format(bow_doc_4310[i][0],
                                               dictionary[bow_doc_4310[i][0]],
bow_doc_4310[i][1]))

from gensim import corpora, models
tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]
from pprint import pprint
for doc in corpus_tfidf:
    pprint(doc)
    break
print(categories)
lda_model = gensim.models.LdaModel(bow_corpus, num_topics=100, id2word=dictionary)
for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

print(bow_corpus[4310],documents['sentence'][4310])
for index, score in sorted(lda_model[bow_corpus[4310]], key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 10)))