from keras.layers import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras import regularizers

import json
import os
import pandas as pd
import numpy as np

from dotenv import load_dotenv
load_dotenv()
datapath = os.environ['DATA_PATH']


def process_image_metadata(X):
    annotations_file = 'image-annotations.json'
    json_data = json.loads(open(annotations_file).read())

    vector_size = len(json_data)
    print(json_data)
    print(X.shape)

    return X


def process_sentiment(X, sentiment_path):
    processedX = X
    # Time to get some sentiment!
    for ind, row in X.iterrows():
        petid = row['PetID']
        sentiment_file = sentiment_path + petid + '.json'
        if os.path.isfile(sentiment_file):
            json_data = json.loads(open(sentiment_file).read())

            processedX.loc[ind,
                           'DescriptionMagnitude'] = json_data['documentSentiment']['magnitude']
            processedX.loc[ind,
                           'DescriptionScore'] = json_data['documentSentiment']['score']
        else:
            processedX.loc[ind, 'DescriptionMagnitude'] = 0
            processedX.loc[ind, 'DescriptionScore'] = 0
    return processedX


def load_input(petfile, sentiment_path, keepPetId=False):
    X = pd.read_csv(petfile)
    X = pd.get_dummies(X, columns=['Type', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3', 'MaturitySize',
                                   'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'State'])

    X = process_sentiment(X, sentiment_path)
    X = process_image_metadata(X)

    X = X.drop(['Description', 'AdoptionSpeed', 'Name', 'RescuerID'],
               axis=1, errors='ignore')

    if keepPetId == False:
        X = X.drop(['PetID'],
                   axis=1, errors='ignore')

    columns_to_normalize = ['Age', 'Quantity', 'Fee',
                            'VideoAmt', 'PhotoAmt', 'DescriptionMagnitude', 'DescriptionScore']

    for column in columns_to_normalize:
        X[column] = (X[column] - X[column].mean()) / X[column].std()
        # X[column] = X[column] / X[column].max()

    X = X.sort_index(axis=1)
    return X


def normalize_output(Y):
    newY = Y / 4

    return newY


def denormalize_output(y):
    ret = (y * 4)
    ret = ret.astype(int)
    return ret


def get_model(input_size):
    model = Sequential()
    model.add(Dense(input_size, input_dim=input_size,
                    activation='relu', kernel_regularizer=regularizers.l1(0.0001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(input_size * 2, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(input_size * 2, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(int(input_size * 1.5), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(input_size, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(input_size, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(input_size, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(input_size, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(input_size, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(5, activation='softmax'))

    return model
