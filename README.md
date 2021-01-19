# Disaster Response Pipeline Project

## Overview

This project is in a form of a Web App implemented using python and its libraries, ETL techniques, Machine Learning Models and Flask App to visualize a system 
that can classify messages sent during disasters and correlate it to various categories in order to facilitate disaster response. 
This is a Udacity Nanodegree Educational Project.

## Contents

The repo consists folders of the following structure:

* app
    * template
        * master.html  # main page of web app
        * go.html  # classification result page of web app
    * run.py  # Flask file that runs app

* data
    * disaster_categories.csv           # data to process 
    * disaster_messages.csv             # data to process
    * process_data.py                   # python script for the ETL Transformation of the data
    * DisasterResponse.db               # database created from code to save cleaned data

* models
    * train_classifier.py               # python script to create machine learning pipeline model for the dataset
    * classifier.pkl                    # saved model

* notebooks
    * ETL_Pipeline_Preparation.ipynb    # Jupyter Notebook to explain ETL Script step by step
    * ML_Pipeline_Preparation.ipynb     # Jupyter Notebook to explain Machine Learning Model creation and improvements step by step

* README.md

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
