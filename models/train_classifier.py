import sys

# import libraries
from sqlalchemy import create_engine 
import nltk
import pandas as pd
import sqlite3
import pickle

import re
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

def load_data(database_filepath):
    
    # load data from database
    engine = create_engine('sqlite:///././'+database_filepath)
    df = pd.read_sql_table('Messages_with_Cats', engine)

    # Choose the X and Y (feature and target variables) for ML Model
    X = df['message']
    Y = df.iloc[:,4:]
    col_names = Y.columns.values
    
    return X, Y, col_names

def tokenize(text):
    
    '''
    INPUT 
        text: Text to be processed   
    OUTPUT
        Returns an array of words after being processed; by being tokenized, lower cased, stripped, and lemmatized
    '''
    
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

    
def build_model():
    
    pipeline = Pipeline([
            ('features', FeatureUnion([

                ('text_pipeline', Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer())
                ]))
            ])),

            ('clf', MultiOutputClassifier(KNeighborsClassifier()))
        ])
    return pipeline


def classificaion_report_iteration(true_y,pred_y,col_names):
    
    '''
    INPUT 
        true_y: Test dataset true results
        pred_y: Predicted results from the test_X in the testing dataset
        col_names: Column Names for each category in order to print its scores
    OUTPUT
        Prints out the scores of each category using the pre-defined function 
        classification_report along which the category name
    '''
    
    for i in range(pred_y[0,:].size):
        print(col_names[i])
        print(classification_report(true_y[i,:],pred_y[i,:]))


def evaluate_model(model, X_test, Y_test, category_names):
    
    # Predict from trained model using the test dataset
    y_pred = model.predict(X_test)
    classificaion_report_iteration(Y_test.values, y_pred, category_names)


def save_model(model, model_filepath):
    
    file = open(model_filepath, 'wb')
    pickle.dump(model, file)
    file.close()

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()