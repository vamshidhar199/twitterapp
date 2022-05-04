# twitterapp
## Overview
- - - 1.Created model for predicting the sentiment of the tweet [training the model has been done in colab].
- - - 2.Saved the model and used flask framework to make predictions.
- - - 3.Everytime user searches something in UI, it actually gets related tweets from the twitter using twitter API and then gives us the details about whether it is negative, positive or neutral speech.
## Technology used
- - 1.Flask Framework
- - 2.HTML 
- - 3.Python

# Description

## Abstract
This document details about collecting the data(tweets) from twitter using twitter API in real time and build a
model that would predict the sentiment of the tweet whether it is positive, negative or neutral. It also describes
about the GaussianNB model used to train the data and the pre-processing that is involved
Finally, it describes about using flask framework to create a user interface to interact with the model and
make predictions on real time tweets.
## Gathering tweets from twitter and Pre-processing
### Twitter API
Twitter provides developer accounts which can be used to access the twitter data through the API’s services
provided we have proper access levels and set of api tockent and secrets. This document uses tweepy to get the
data from twitter.

## Labeling the data
### Using textblob

Since the data acquired from the twitter is not labeled, text blob has been used to label the data based on the
polarity of the tweet into positive, negative and neutral speech. To verify the resultant labels, the csv has been
manually checked for the correctness of the labeling.
### Cleaning the data
In order to remove certain patterns which are not alphabets and to remove some unnecessary characters, the
data has been processed to make the tweets free from these unwanted characters.


### Lemmatization
Lemmatization is the process of grouping together the different inflected forms of a word so they can be analyzed
as a single item. Lemmatization is similar to stemming but it brings context to the words. So it links words
with similar meanings to one word. 

### Bag of words and TF-IDF
The bag-of-words model is a way of representing text data when modeling text with machine learning algorithms.
The bag-of-words model is simple to understand and implement and has seen great success in problems such as
language modeling and document classification. Also used TF-IDF as it is proven to provide better results compared to BOW.

## Modeling the data
### GaussianNB
Naive Bayes is a basic but effective probabilistic classification model in machine learning that draws influence
from Bayes Theorem. Bayes theorem is a formula that offers a conditional probability of an event A taking
happening given another event B has previously happened. Gaussian Naive Bayes classifier is employed when
the predictor values are continuous and are expected to follow a Gaussian distribution.

Trained the model with both the bag of words and tf-idf and the reason being that the stop words might
not have been complete removed in bag of words approach and in tf-idf , idf removes the unwanted words which
makes it more cleaner and help model train with relevant data.
### Tensor flow based model - LSTM
We start with defining the input layer, and next is embedding. Embeddings provide the presentation of words
and their relative meanings. Like in this, we are feeding the limit of maximum words, lenght of input words
and the inputs of previous layer. LSTM (long short term memory) save the words and predict the next words
based on the previous words. LSTM is a sequance predictor of next coming words.
By receiving inputs from the Faltten layer, the Dense layer reduces the outputs. The dense layer takes all
of the previous layer’s inputs, does calculations, and sends 256 outputs.The activation function is a node that
is placed at the end of or in between the layers of a neural network model. The activation function assists in
determining which neurons should be passed and which should fire. So, given an input or a group of inputs, the
activation function of a node defines the output of that node. Finally the drop out layers prevent over fitting.

## Downloading the created model
dump ( naive_classifier , filename =" text_classification . joblib ")
The model which has been trained in the previous step has been stored and downloaded from colab so that it
can be used with flask and integrated to a UI (web application)

## Model Comparison
![alt](https://github.com/vamshidhar199/twitterapp/blob/master/bonusComparison.png)
## Flask Framework for UI Integrations
Flask is a micro web framework written in Python. It is classified as a micro framework because it does not
require particular tools or libraries. Flask is a powerful framework and is one of the most commonly used
framework for ML models integration with the user interface.
The model from the colab has been imported to the flask project structure and has been integrated using the
following code. The home.py file has the code to call the model with the tweet data which has been gathered
from the get-related-tweets method which returns the relevant tweets from the twitter in real time as twitter
would return if we search in the twitter search box. This data is then pre processed before being sent to the
model for prediction.
A virtual environment has been created with name twitter ans installed all the necessary packages required
for the running of the application.
The templates folder contains the code for HTML which basically displays the user screens.The folder
structure is as shown below.



The detailed code can be found at the git hub repository mentioned at the beginning of the document.
The UI screens would look as shown below.

![alt](https://github.com/vamshidhar199/twitterapp/blob/master/Screen%20Shot%202022-05-02%20at%201.50.04%20PM.png)
![alt](https://github.com/vamshidhar199/twitterapp/blob/master/Screen%20Shot%202022-05-01%20at%203.59.39%20PM.png)
![alt](https://github.com/vamshidhar199/twitterapp/blob/master/Screen%20Shot%202022-05-01%20at%204.00.16%20PM.png)

