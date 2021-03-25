# Sentiment-analysis

The goal of this project is to analyse the tweet and predict the sentiment accordingly.
Sentiment analysis can be done in various ways the, but this project focuses on twitter 
data.
The project mainly addresses the sentiment of the given text data using natural 
language processing and machine learning algorithm we try to predict the sentiment 
of the twitter text data.
The project mainly composes of machine learning algorithms like Logistic 
Regression, Support vector classifier, Naïve Base algorithm, and deep learning 
algorithm like LSTM
By implementation of the algorithms we try to predict the sentiment of the tweet 
supplied by the user using Flask implementation of the tweet mood-based analysis.
Displaying the positive mood according to the tweet and negative mood accordingly.

## STEPS INVOLVED

### Data Pre-processing -

Data pre-processing is necessary step before doing sentiment analysis as there can be
punctuations, unrelated words, duplicate words, etc. which does not give an accurate 
result. so, there are several processes which should be undertaken before doing the 
analysis.

### Stop words removal: -

Stop words refers to the most common words in a sentence. The stop words include 
in, is, should, were, was, the, etc. which doesn’t have any impact on the analysis and it 
is not important even if we remove it. So, such unmeaningful words need to be
removed. Python has an inbuilt dictionary of stop words which includes many 
common stop words and also you can update it if the words are not included in it.

### Tokenization: -

There are 2 types of token sentence token and word token. When the paragraph gets 
divided into sentences it is called sentence token and when the sentences gets divided 
into words it is called as words token. Normally the sentences get divided into words 
so that it’s easier to pull out the important features from the sentence.

### Stemming: -

Stemming is the process of reducing the words to its base word
for e.g. Knowingly is converted to know so such transformation/stemming of words 
helps for better accuracy.

### Lemmatization: -

It is the process of reducing the inflectional ending words to its base word but for 
lemmatizing to work properly we need to check that there are more similar words 
with different forms such that
one word remains and the rest are eliminated. Which also take care of duplication of 
words.

### Removing punctuations: -

Punctuations plays a major role when a sentence is formed but for sentimental 
analysis it is of no use so it needs to be removed before proceeding with the analysis.
Sometimes even the symbols are removed.

### Addressing the missing data: -

Missing data should be taken care of whenever some analysis is carried out.
The pre-processing part related to this project is as follows:
1. Lower Casing: Each text is converted to lowercase.

2. Replacing URLs: Links starting with "http" or "https" or "www" are replaced 
by " " empty space.

3. Replacing Emojis: Replace emojis by using a pre-defined dictionary 
containing emojis along with their meaning. (eg: ":)" to "EMOJIsmile")

4. Replacing Usernames: Replace @Usernames with word “ ”

5. Removing Non-Alphabets: Replacing characters except Digits and Alphabets 
with a space.

6. Removing Consecutive letters: 3 or more consecutive letters are replaced by 2 letters. (eg: "Heyyyy" to "Heyy")

7. Removing Short Words: Words with length less than 2 are removed.

8. (Case 1)Removing Stopwords: Stopwords are the English words which does 
not add much meaning to a sentence. They can safely be ignored without 
sacrificing the meaning of the sentence. (eg: "the", "he", "have")

  (Case 2)Not removing Stopwords: Word importance may vary depending on 
  the dataset but some time not removing stopword can also help in good 
  prediction and accuracy results.

9. Lemmatizing: Lemmatization is the process of converting a word to its base 
form. (e.g: “Great” to “Good”)



## Feature extraction: 

### TF-IDF: 
-TF-IDF is better and accurate technique which is carried out to extract the 
importance of the words to a document in a collection or corpus. Here TF is called as
Term Frequency and IDF is called as Inverse Document Frequency. Term is nothing 
but the words or phrases in a document. The importance of the words can be 
determined by finding out the frequency of that particular word so the importance is 
proportional to the number of times the word is occurring in the document. This 
method is very useful for text mining and used for retrieving the information.
In order to interpret the text or data collection, TF-IDF shows how relevant the term 
is. Let's take an example to learn. Suppose you have a dataset where students write an 
essay on the subject, My Home. In this dataset the word an always appears; in contrast 
with other terms in the dataset, this is a high-frequency word. The dataset includes 
other terms such as home, residence, quarters, and so on, which are less frequently 
seen, because their prevalence is lower and they provide more detail than the term. 
This is TF-IDF 's experience.
TF-IDF Vectoriser: TF-IDF Vectoriser transforms a set of raw documents to a TFIDF matrix. The vectoriser is normally trained on the X train dataset.


### Ngram:
N-grams of texts are extensively used in text mining and natural language processing 
tasks. An n-gram is a contiguous sequence of n items from a given sample of text or 
speech. an n-gram of size 1 is referred to as a "unigram"; size 2 is a 
"bigram"; size 3 is a "trigram". When N>3 this is usually referred to as four grams 
or five grams and so on.
Formula to calculate number of N-grams in a sentence.
If X=Number of words in a given sentence K, the number of n-grams for sentence K
would be:
### Unigram:
Ngram k = X - (N - 1)
Example:
Sentence : I want to learn Machine Learning


Unigram: now calculate number of unigrams in sentence using formula
here, X = 6 and N = 1 (for unigram)
Ngram k = X - (N - 1)
Ngram k = 6 - (1–1) = 6 (i.e. unigram is equal to number of words in a sentence)
[I][want][to][learn][Machine][Learning]


### Biagram:
here, X = 6 and N = 2 (for biagram)
Ngram k = X - (N - 1)
Ngram k = 6 - (2–1) = 5
[I want][want to][to learn][learn Machine][Machine Learning]

### Trigram:
here, X = 6 and N = 3 (for trigram)
Ngram k = X - (N - 1)
Ngram k = 6 - (3–1) = 4
[I want to][want to learn][to learn Machine][learn Machine Learning]
You can also generate for N=4,5,6 and so on.
  
## Machine learning 
Algorithm implemented are Logistic Regression, Support Vector Classifier, BNBBernoulliNB, AdaBoostClassifier, RandomForest, Decision Tree, Multi-layer Perceptron
   
## Deep learning
Algorithm implemented is Long Short-Term Memory (LSTM)


## Following Are the ScreenShot of the application

Main Page

![image](https://github.com/BhavyaShukla/Sentiment-analysis/blob/main/screenshot/Capture.PNG)

Positively Classified

![image](https://github.com/BhavyaShukla/Sentiment-analysis/blob/main/screenshot/Positive.PNG)

Negatively Classified

![image](https://github.com/BhavyaShukla/Sentiment-analysis/blob/main/screenshot/Negative.PNG)

Deep Learning Positively Classified

![image](https://github.com/BhavyaShukla/Sentiment-analysis/blob/main/screenshot/Deep%20learning%20pos.PNG)

Deep learning Negatively Classified

![image](https://github.com/BhavyaShukla/Sentiment-analysis/blob/main/screenshot/Deep%20learning%20Negative.PNG)

