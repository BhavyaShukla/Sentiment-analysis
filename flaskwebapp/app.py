from flask import Flask,render_template,request,url_for,redirect
# utilities
import re
import pickle
import numpy as np
import pandas as pd
import nltk
from tensorflow.keras.models import load_model
import tensorflow as tf
import joblib
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
app=Flask(__name__,template_folder='html',static_folder="static")
def loadtoken():
    token = joblib.load('tokenizer.pkl')
    return token

    return token
def load_tensfl(d):
    with tf.device('/GPU:0'):
        model1 = load_model('modellstm2.h5')
        score=model1.predict(d)
    return score



def load_word2vec():
    word2vec = joblib.load('word2vec.pkl')
    return word2vec
# Defining dictionary containing all emojis with their meanings.
emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad',
          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed',
          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink',
          ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}

## Defining set containing all stopwords in english.
stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
             'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
             'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',
             'does', 'doing', 'down', 'during', 'each','few', 'for', 'from',
             'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
             'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
             'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
             'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
             'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're',
             's', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
             't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
             'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
             'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
             'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
             'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",
             "youve", 'your', 'yours', 'yourself', 'yourselves']

def listToString(s):

    # initialize an empty string
    str1 = " "

    # return string
    return (str1.join(s))
def preprocess1(text):
    review=re.sub('@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+',' ',text)
    review=review.lower()
    review=review.split()
    review=[word for word in review if not word in stopwordlist]
    print(review)

    token=loadtoken()
    review1=pad_sequences(token.texts_to_sequences([review]), maxlen=300)

    #print(vocab_size)
    print(review)
    return review,review1


def preprocess(textdata):
    processedText = []

    # Create Lemmatizer and Stemmer.
    wordLemm = WordNetLemmatizer()

    # Defining regex patterns.
    urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    userPattern       = '@[^\s]+'
    alphaPattern      = "[^a-zA-Z0-9]"
    sequencePattern   = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"

    for tweet in textdata:
        tweet = tweet.lower()

        # Replace all URls with 'URL'
        tweet = re.sub(urlPattern,' URL',tweet)
        # Replace all emojis.
        for emoji in emojis.keys():
            tweet = tweet.replace(emoji, "EMOJI" + emojis[emoji])
        # Replace @USERNAME to 'USER'.
        tweet = re.sub(userPattern,' USER', tweet)
        # Replace all non alphabets.
        tweet = re.sub(alphaPattern, " ", tweet)
        # Replace 3 or more consecutive letters by 2 letter.
        tweet = re.sub(sequencePattern, seqReplacePattern, tweet)
        tweetwords = ''
        for word in tweet.split():
            # Checking if the word is a stopword.
            #if word not in stopwordlist:
            if len(word)>1:
                # Lemmatizing the word.
                word = wordLemm.lemmatize(word)
                tweetwords += (word+' ')

        processedText.append(tweetwords)

    return processedText






def predict(vectoriser, model, text):
    # Predict the sentiment
    textdata = vectoriser.transform(preprocess(text))
    sentiment = model.predict(textdata)

    # Make a list of text with sentiment.
    data = []
    for text, pred in zip(text, sentiment):
        data.append((text,pred))

    # Convert the list into a Pandas DataFrame.
    df = pd.DataFrame(data, columns = ['text','sentiment'])
    df = df.replace([0,1], ["Negative","Positive"])
    return sentiment
def load_models():

    # Load the vectoriser.
    file = open('html/vectoriser-ngram-(1,2).pickle', 'rb')
    vectoriser = pickle.load(file)
    file.close()
     # Load the BNB Model.
    file = open('html/Sentiment-LR-ngram-(1,2).pickle', 'rb')
    LR = pickle.load(file)
    file.close()

    return vectoriser, LR
@app.route('/',methods=["POST","GET"])
def home():
    if request.method=='POST':
        vectoriser, LR = load_models()
        tweet1=request.form.get('tweet')
        text1=[]
        text1.append(tweet1)
        # print(text1)
        pr = predict(vectoriser,LR, text1)
        sentipos="Positive"
        sentineg="Negative"
        senti=""
        s1=pr[0]
        if s1 == 0:
            senti=sentineg
        else:
            senti=sentipos
        # print(pr)
        return render_template('index.html',tweet=senti,s1=s1)
    else:
        return render_template('index.html')
@app.route('/classify',methods=["POST","GET"])
def classify():

    # word2vec = joblib.load('word2vec.pkl')
    if request.method=='POST':
        res1=request.form.get('res')

        d,e=preprocess1(res1)
        m1=load_tensfl(e)
        m=m1[0]

        for i in m:
            if i>0.50:
                senti="Positive"
                s1=1
            elif i<0.50:
                senti="Negative"
                s1=0
        #print(m1)
        # score=model1.predict(d)
        # score=score[0]


        return render_template('classify.html',res=s1,tweet=senti,score=m,word=d)
    else:
        return render_template('classify.html')
if __name__ == "__main__":

    app.run(debug=True)
