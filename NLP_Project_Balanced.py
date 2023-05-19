#NLP GROUP PROJECT CODE --------------------------------------------------------------------------------------
#Harith Izani 1821037
#Muhammad Hariz Bin Hasnan 1827929
#Mohamad Arif Daniel Bin Muhamaddun 1917027
#Nur Atiqah binti Hasbullah 1920744

#IMPORTING INITIAL PACKAGES AND MISC -------------------------------------------------------------------------
import nltk
import numpy as np
import pandas as pd

nltk.download("movie_reviews", "twitter_samples")

#PREPARING DATASET -------------------------------------------------------------------------------------------
dataset = []
#From movie reviews dataset index 0-1
from nltk.corpus import movie_reviews
dataset.append([nltk.corpus.movie_reviews.raw(review) for review in nltk.corpus.movie_reviews.fileids(categories=["pos"])])
dataset.append([nltk.corpus.movie_reviews.raw(review) for review in nltk.corpus.movie_reviews.fileids(categories=["neg"])])
#From twitter samples dataset index 2-3
from nltk.corpus import twitter_samples
dataset.append(twitter_samples.strings('positive_tweets.json'))
dataset.append(twitter_samples.strings('negative_tweets.json'))

#Train-Test-Split
import math
dataset_pos = dataset[0] + dataset[2][:1000]
dataset_neg = dataset[1] + dataset[3][:1000]
pos_len = math.floor(len(dataset_pos)/4)
neg_len = math.floor(len(dataset_neg)/4)

test_pos = dataset_pos[:pos_len]
train_pos = dataset_pos[pos_len:]
test_neg = dataset_neg[:neg_len]
train_neg = dataset_neg[neg_len:]

train_x = train_pos + train_neg 
test_x = test_pos + test_neg

train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis=0)
test_y = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)), axis=0)

#PREPROCESSING -----------------------------------------------------------------------------------------------
#Initializing stopwords + names
stop_words = nltk.corpus.stopwords.words("english")
stop_words.extend([w.lower() for w in nltk.corpus.names.words()])

#Processing Text Function
import re
import string
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
def process_text(sentence):
   
    stemmer = PorterStemmer()
    
    # remove stock market tickers like $GE
    sentence = re.sub(r'\$\w*', '', sentence)
    
    # remove old style retweet text "RT"
    sentence = re.sub(r'^RT[\s]+', '', sentence)
    
    # remove hyperlinks
    sentence = re.sub(r'https?:\/\/.*[\r\n]*', '', sentence)
    
    # remove hashtags
    # only removing the hash # sign from the word
    sentence = re.sub(r'#', '', sentence)
    
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    sentence_tokens = tokenizer.tokenize(sentence) #using tweet tokenizer to keep hashtag

    sentence_clean = []
    for word in sentence_tokens:
        if (word not in stop_words and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            # tweets_clean.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            sentence_clean.append(stem_word)

    return sentence_clean

#Extracting All Words from Dataset (turning dataset of texts into dataset of words)
def extract_words(text):
    word_list = []
    for sentence in text:

        word_list.extend(process_text(sentence))
    return word_list

for num,text in enumerate(dataset):
    dataset[num] = extract_words(text)

#Wordcount function
def word_count(ds):
    count = 0
    for datasets in ds:
        count += len(datasets)
    return count #return wordcount for list of lists

#Removing Stop Words and Names
print("Word count pre-filtering", word_count(dataset))
def word_filter(worldlist):
    temp_array = []
    for w in worldlist:
        if (w not in stop_words): #skips stopwords
            temp_array.append(w)
    return temp_array

for num,datasets in enumerate(dataset):
    dataset[num] = word_filter(datasets)

#Removing duplicates
def remove_duplicates(ds): #splits dataset into 1/4 then removes duplicates by batch
    ds_length = math.floor(len(ds)/4)
    temp_holder = []
    temp_holder = list(dict.fromkeys(ds[:ds_length]))
    temp_holder.extend(list(dict.fromkeys(ds[ds_length:(ds_length+ds_length)])))
    temp_holder.extend(list(dict.fromkeys(ds[(ds_length+ds_length):])))
    return temp_holder
"""
for num in range(len(dataset)):
    dataset[num] = remove_duplicates(dataset[num])"""
print("Word count post-filtering", word_count(dataset))

#Dataset Labeling for Freq Table
dataset_pos = dataset[0] + dataset[2]
dataset_neg = dataset[1] + dataset[3]
labels = np.append(np.ones((len(dataset_pos))), np.zeros((len(dataset_neg))))

#Building Frequency Table (a dictionary of {(word, polarity): freq})
def build_freqs(ds, lbl):
    freqs = {}
    pairings = zip(ds, lbl)
    for word,sentiment in pairings:
        pair = (word, sentiment)
        if pair in freqs:
            freqs[pair] += 1
        else:
            freqs[pair] = 1
    return freqs

freqs = build_freqs(dataset_pos, labels[:len(dataset_pos)])
freqs.update(build_freqs(dataset_neg, labels[len(dataset_pos):]))

#MODEL DEVELOPMENT & TRAINING ---------------------------------------------------------------------------------
def sigmoid(z): 

    h = 1/(1+np.exp(-z))
   
    return h
def gradientDescent(x, y, theta, alpha, num_iters):
   
    m = x.shape[0]
    
    for i in range(0, num_iters):
        
        # get z, the dot product of x and theta
        z = np.dot(x,theta)
        
        # get the sigmoid of z
        h = sigmoid(z)
        
        # calculate the cost function
        J = -(np.dot(y.T,np.log(h))+np.dot((1-y).T,np.log(1-h)))/m

        # update the weights theta
        theta = theta - alpha*(np.dot(x.T,h-y))/m
        
    J = float(J)
    return J, theta

#Feature Extractor for Model
def extract_features(text, freqs):
    
    # process_tweet tokenizes, stems, and removes stopwords
    word_l = process_text(text)
    
    # 3 elements in the form of a 1 x 3 vector
    x = np.zeros((1, 3)) 
    
    #bias term is set to 1
    x[0,0] = 1 
       
    # loop through each word in the list of words
    for word in word_l:
         
        if (word,1.0) in freqs:
            # increment the word count for the positive label 1
            x[0,1] += freqs[(word,1.0)]
        if(word,0.0) in freqs:
            # increment the word count for the negative label 0
            x[0,2] += freqs[(word,0.0)]
        
    return x

#Training Model
X = np.zeros((len(train_x), 3)) #collect the features 'x' and stack them into a matrix 'X'
for i in range(len(train_x)):
    X[i, :]= extract_features(train_x[i], freqs)
Y = train_y #training labels corresponding to X

print(X.shape)
print(Y.shape)

#Gradient Descent application
J, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-9, 1500)

#Show performance metric
print(f"The cost after training is {J:.6f}.")
print(f"The resulting vector of weights is {[round(t, 8) for t in np.squeeze(theta)]}")

#VISUALIZATION -------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt

def word_location(text_range, freqs):
    keys = []
    for text in text_range:
        keys.extend(process_text(text))
    
    data = []

    for word in keys:
        pos = 0
        neg = 0

        if (word, 1) in freqs:
            pos = freqs[(word,1)]
        if (word,0) in freqs:
            neg = freqs[(word,0)]

        data.append([word, pos, neg])

    return data

def plot_word_location(data):
    fig, ax = plt.subplots(figsize = (8, 8))

    # convert positive raw counts to logarithmic scale. we add 1 to avoid log(0)
    x = np.log([x[1] + 1 for x in data])  

    # do the same for the negative counts
    y = np.log([x[2] + 1 for x in data]) 

    # Plot a dot for each pair of words
    ax.scatter(x, y)  

    # assign axis labels
    plt.xlabel("Log Positive count")
    plt.ylabel("Log Negative count")

    # Add the word as the label at the same position as you added the points just before
    for i in range(0, len(data)):
        ax.annotate(data[i][0], (x[i], y[i]), fontsize=12)

    ax.plot([0, 9], [0, 9], color = 'red')  # Plot the red line that divides the 2 areas.
    plt.show()

def neg(theta, pos):
    return (-theta[0] - pos * theta[1]) / theta[2]

fig, ax = plt.subplots(figsize = (10, 8))

colors = ['red', 'green']

# Color base on the sentiment Y
ax.scatter(X[:,1], X[:,2], c=[colors[int(k)] for k in Y], s = 0.1)  # Plot a dot for each pair of words
plt.xlabel("Positive")
plt.ylabel("Negative")

# Now lets represent the logistic regression model in this chart. 
maxpos = np.max(X[:,1])           # max value in x-axis

# Plot a gray line that divides the 2 areas.
ax.plot([0,  maxpos], [neg(theta, 0),   neg(theta, maxpos)], color = 'gray') 

plt.show()

#TESTING -------------------------------------------------------------------------------------------------------
def predict_text(text, freqs, theta):

    # extract the features of the tweet and store it into x
    x = extract_features(text, freqs)
    
    # make the prediction using x and theta
    y_pred = sigmoid(np.dot(x,theta))
    
    return y_pred

def testing(text):
    y_pred = predict_text(text, freqs, theta)
    
    if y_pred > 0.5:
        print('Positive sentiment')
    else: 
        print('Negative sentiment')

from sklearn.metrics import classification_report, confusion_matrix
def test_logistic_regression(test_x, test_y, freqs, theta):
    y_hat = []
    
    for text in test_x:
        # get the label prediction for the text
        y_pred = predict_text(text, freqs, theta)
        
        if y_pred > 0.5:
            y_hat.append(1.0)
        else:
            y_hat.append(0.0)

    accuracy = np.sum(np.squeeze(test_y) == np.squeeze(np.asarray(y_hat)))/len(test_y)

    return accuracy, y_hat

test_accuracy,y_hat = test_logistic_regression(test_x, test_y, freqs, theta)
print(f"Logistic regression model's accuracy = {test_accuracy:.4f}")
print(confusion_matrix(test_y,y_hat))
print(classification_report(test_y,y_hat))
