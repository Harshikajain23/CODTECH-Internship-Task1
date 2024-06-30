import random
from flask import Flask, render_template, request, flash, url_for, jsonify
import nltk
import numpy as np
import pandas as pd
import re
import os
import tensorflow as tf
from numpy import array
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy as nltk_accuracy
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from heapq import nlargest

app = Flask(__name__)

def extract_features(words):
    return dict([(word, True) for word in words])

def train_sentiment_classifier():
    nltk.download('movie_reviews')
    nltk.download('punkt')
    nltk.download('vader_lexicon')
    nltk.download('stopwords')

    # Load the movie reviews dataset
    movie_reviews_categories = movie_reviews.categories()
    documents = [(list(movie_reviews.words(fileid)), category)
                 for category in movie_reviews_categories
                 for fileid in movie_reviews.fileids(category)]
    random.shuffle(documents)

    # Split the dataset into training and testing sets
    training_documents = documents[:1600]
    testing_documents = documents[1600:]

    # Extract features from the training dataset
    training_features = [(extract_features(d), c) for (d, c) in training_documents]
    testing_features = [(extract_features(d), c) for (d, c) in testing_documents]

    # Train a Naive Bayes classifier
    classifier = NaiveBayesClassifier.train(training_features)

    # Evaluate the classifier
    print(f"Accuracy: {nltk_accuracy(classifier, testing_features) * 100:.2f}%")

    return classifier

classifier = train_sentiment_classifier()

def analyze_sentiment(text, classifier):
    # Using TextBlob
    blob = TextBlob(text)
    if blob.sentiment.polarity > 0:
        textblob_sentiment = "positive"
    elif blob.sentiment.polarity < 0:
        textblob_sentiment = "negative"
    else:
        textblob_sentiment = "neutral"

    # Using VADER
    vader = SentimentIntensityAnalyzer()
    vader_scores = vader.polarity_scores(text)
    vader_percentages ={
        "positive": vader_scores['pos'] * 100,
        "neutral": vader_scores['neu'] * 100,
        "negative": vader_scores['neg'] * 100,
        "compound": vader_scores['compound'] * 100
    }
    max_vader_percentage = max(vader_percentages.values())
    vader_sentiment = max(vader_scores, key=vader_scores.get)
    
    

    # Using the Naive Bayes classifier
    # words = text.split()
    # features = extract_features(words)
    # nb_sentiment = classifier.classify(features)

    return {
        "textblob sentiment": textblob_sentiment,
        "vader sentiment": vader_sentiment,
        "vader sentiment-percentage": max_vader_percentage
        
        # "naive_bayes_sentiment": nb_sentiment
    }

def summarize_text(raw_text):
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(raw_text.lower())
    word_freq = FreqDist(words)

    # Assign score to each sentence based on word frequency
    sentences = sent_tokenize(raw_text)
    sentence_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in word_freq and word not in stop_words:
                if sentence not in sentence_scores:
                    sentence_scores[sentence] = word_freq[word]
                else:
                    sentence_scores[sentence] += word_freq[word]

    # Select top 5 sentences with highest scores
    summarized_sentences = nlargest(5, sentence_scores, key=sentence_scores.get)
    summarized_text = ' '.join(summarized_sentences)
    return summarized_text

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'analyze_text' in request.form:
            sample_text = request.form['text']
            sentiment_result = analyze_sentiment(sample_text, classifier)
            return jsonify(sentiment_result)
        elif 'summarize_text' in request.form:
            raw_text = request.form['text']
            summary_result = summarize_text(raw_text)
            return jsonify({'summary': summary_result})
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
