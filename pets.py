from keras.layers import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras import regularizers

import json
import os
import pandas as pd
import numpy as np


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
    # newY = pd.DataFrame([np.ones(x)
    #                      for x in Y['AdoptionSpeed']]).fillna(0).astype(int)

    return newY


def denormalize_output(y):
    # output_values = np.array(
    #     [[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]])
    # ret = [np.argmin(np.linalg.norm(output_values-i, axis=1)) for i in y]
    ret = np.rint(y * 4)
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

    model.add(Dense(1, activation='sigmoid'))

    return model
