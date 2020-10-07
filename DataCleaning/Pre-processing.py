# ## Cleaning Data
# ### Team Innovation Geeks

import pandas as pd
import numpy as np
df = pd.read_csv('https://data.ca.gov/dataset/ae343670-f827-4bc8-9d44-2af937d60190/resource/bb82edc5-9c78-44e2-8947-68ece26197c5/download/purchase-order-data-2012-2015-.csv')

# ### Data Preprocessing

df.astype({"Item Name":'str', 'Item Description': 'str'}) # changing data types so clean_text function works properly
df.dropna(subset=['Item Description'], how = 'any', inplace=True) # removed rows with null values in 'Item Description' column
df.dropna(subset=['Classification Codes'], how = 'any', inplace=True) # removed rows with null values in 'Classification Codes' column
#df.dropna(subset=['Class'], how = 'any', inplace=True) # removed rows with null values in 'Class' column ### not sure if want to do this as well
df.isnull().sum()


# import libraries
import nltk
nltk.download('words')
nltk.download('stopwords')
import re
import string as st
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer # or LancasterStemmer, RegexpStemmer, SnowballStemmer
words = set(nltk.corpus.words.words())

# create clean_text function:
def clean_text(text):
    
    def simplify_punctuation(text):
        text = str(text)
        text = re.sub(r'[^a-zA-Z0-9]', ' ', text) # replace everything except letters/numbers
        text = re.sub(r'([\w])\1+', r'\1', text) # reduce exaggeration
        return text
    
    def tokenize(text):
        text = re.split('\s+' ,text)
        return [x.lower() for x in text]
    
    def remove_stopwords(text):
        return [word for word in text if word not in nltk.corpus.stopwords.words('english')]
        
    def stemming(text):
        ps = PorterStemmer()
        return [ps.stem(word) for word in text]

    def lemmatize(text):
        word_net = WordNetLemmatizer()
        return [word_net.lemmatize(word) for word in text]
    
    def normalize_whitespace(text):
        text = str(text)
        text = re.sub(r"//t", r"\t", text)
        text = re.sub(r"( )\1+", r"\1", text)
        text = re.sub(r"(\n)\1+", r"\1", text)
        text = re.sub(r"(\r)\1+", r"\1", text)
        text = re.sub(r"(\t)\1+", r"\1", text)
        return text.strip(" ")

    text = simplify_punctuation(text) # remove punctuation
    text = tokenize(text) # tokenize
    text = remove_stopwords(text) # remove stopwords
    #text = stemming(text) # stemming
    #text = lemmatize(text) # lemmatization, not used because words are stemmed. Could use in combination with stemming but didn't think it was necessary. 
    #text = normalize_whitespace(text) # remove extra white space. This impacts tokenization which is why I don't have it running. 

    return text


# ### Apply Cleaning Function to Columns

df['clean_title'] = df['Item Name'].apply(lambda x : clean_text(x))
df['clean_desc'] = df['Item Description'].apply(lambda x : clean_text(x))
df['clean_title_desc'] = df['clean_title'] + df['clean_desc']


## removing tokens to create a sentence
def return_sentences(tokens):
    return " ".join([word for word in tokens])

df['clean_title_sent'] = df['clean_title'].apply(lambda x : return_sentences(x))
df['clean_desc_sent'] = df['clean_desc'].apply(lambda x : return_sentences(x))
df['clean_title_desc_sent'] = df['clean_title_sent'] + " " + df['clean_desc_sent']
df.head()


# #### Write Clean Data to New CSV
df.to_csv('cleaned_data.csv')

