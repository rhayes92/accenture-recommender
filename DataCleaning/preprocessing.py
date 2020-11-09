# ## Cleaning Data
# ### Team Innovation Geeks
import boto3
import botocore 
import pandas as pd 
import numpy as np
from sagemaker import get_execution_role 

role = get_execution_role() 

bucket = 'INSERT_BUCKET_NAME_HERE' 
data_key = 'INSERT_FILE_NAME_HERE.csv'
data_location = 's3://{}/{}'.format(bucket, data_key) 

df = pd.read_csv(data_location)

# ### Data Preprocessing
df.astype({"ORDER_TITLE":'str', 'LINE_DESCRIPTION': 'str'}) # changing data types so clean_text function works properly
df.dropna(subset=['ORDER_TITLE'], how = 'any', inplace=True) # removed rows with null values in 'ORDER_TITLE' column 
df.dropna(subset=['LINE_DESCRIPTION'], how = 'any', inplace=True) # removed rows with null values in 'LINE_DESCRIPTION' column
df.dropna(subset=['PSC_CODE'], how = 'any', inplace=True) # removed rows with null values in 'PSC_CODE' column
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
        return text
    
    def tokenize(text):
        text = re.split('\s+' ,text)
        return [x.lower() for x in text]
    
    def remove_stopwords(text):
        return [word for word in text if word not in nltk.corpus.stopwords.words('english')]
        
    #def stemming(text):
        #ps = PorterStemmer()
        #return [ps.stem(word) for word in text]

    #def lemmatize(text):
        #word_net = WordNetLemmatizer()
        #return [word_net.lemmatize(word) for word in text]

    text = simplify_punctuation(text) # remove punctuation
    text = tokenize(text) # tokenize
    text = remove_stopwords(text) # remove stopwords
    #text = stemming(text) # stemming, not used. Affected the ability for the word embedding layer to read words. 
    #text = lemmatize(text) # lemmatization, not used. Affected the ability for the word embedding layer to read words. 
    return text


# ### Apply Cleaning Function to Columns

df['clean_title'] = df['ORDER_TITLE'].apply(lambda x : clean_text(x)) # creating new column with the clean order title
df['clean_desc'] = df['LINE_DESCRIPTION'].apply(lambda x : clean_text(x)) # creating new column with the clean line description
df['clean_title_desc'] = df['clean_title'] + df['clean_desc'] # concatenating the clean order title and clean description to use as x value in the model
df.head()

## removing tokens to return to sentence
def return_sentences(tokens):
    return " ".join([word for word in tokens])

df['clean_title'] = df['clean_title'].apply(lambda x : return_sentences(x))
df['clean_desc'] = df['clean_desc'].apply(lambda x : return_sentences(x))
df['clean_title_desc'] = df['clean_title'] + " " + df['clean_desc']

# removing words with 3 or less characters - this improved the accuracy of the model
df['clean_title'] = df['clean_title'].replace(r'\b\w{1,3}\b', "", regex=True)
df['clean_desc'] = df['clean_desc'].replace(r'\b\w{1,3}\b', "", regex=True)
df['clean_title_desc'] = df['clean_title_desc'].replace(r'\b\w{1,3}\b', "", regex=True)


# #### Write Clean Data to New CSV
df.to_csv('cleaned_data.csv')

# uploading csv file to S3 bucket
s3 = boto3.resource('s3')
s3.meta.client.upload_file('FILE_NAME_HERE.csv', bucket, 'FILE_NAME_HERE.csv')

