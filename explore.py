from dotenv import load_dotenv
from keras.callbacks import CSVLogger
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.utils import shuffle
from tensorflow.python.client import device_lib
import json
import kappa
import matplotlib.pyplot as plt
import os
import pandas as pd
import time
load_dotenv()

datapath = os.environ['DATA_PATH']

X = pd.read_csv(datapath + '/train.csv')
Y = pd.read_csv(datapath + '/train.csv', header=0, usecols=['AdoptionSpeed'])
Y = pd.get_dummies(Y['AdoptionSpeed'], columns=['AdoptionSpeed'])

X = pd.get_dummies(X, columns=['Type', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3', 'MaturitySize',
                               'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'State', 'RescuerID'])

X['DescriptionLength'] = X['Description'].str.len()


# Time to get some sentiment!
for ind, row in X.iterrows():
    petid = row['PetID']
    sentiment_file = datapath + '/train_sentiment/' + petid + '.json'
    if os.path.isfile(sentiment_file):
        json_data = json.loads(open(sentiment_file).read())

        X.loc[ind, 'DescriptionMagnitude'] = json_data['documentSentiment']['magnitude']
        X.loc[ind, 'DescriptionScore'] = json_data['documentSentiment']['score']
    else:
        X.loc[ind, 'DescriptionMagnitude'] = 0
        X.loc[ind, 'DescriptionScore'] = 0

X = X.drop(['Description', 'AdoptionSpeed', 'Name', 'PetID'], axis=1)

columns_to_normalize = ['DescriptionLength', 'Age',
                        'Quantity', 'Fee', 'VideoAmt', 'PhotoAmt']

for column in columns_to_normalize:
    X[column] = (X[column] - X[column].mean())/X[column].std()

X = shuffle(X)

input_units = X.shape[1]
output_units = Y.shape[1]

model = Sequential()
model.add(Dense(input_units, input_dim=input_units, activation='relu'))

model.add(Dense(input_units * 2, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(input_units, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(input_units, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(input_units, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(output_units, activation='softmax'))

adam = Adam(lr=0.1, decay=0.0001)
model_name = time.strftime('%Y-%m-%d-%H-%M-%S')

model.compile(loss='categorical_crossentropy',
              optimizer=adam, metrics=['accuracy'])

csvLogger = CSVLogger('data/' + model_name + '.csv')
history = model.fit(X, Y, epochs=250, shuffle=True, batch_size=1000,
                    validation_split=0.05, verbose=1, callbacks=[csvLogger])

scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

model.save('models/' + model_name + '.h5')

plt.subplot(121)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model acc')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train_acc', 'test_acc'], loc='upper left')
# plt.show()
# plt.savefig('graphs/val' + model_name + '.png')

# summarize history for loss
plt.subplot(122)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss', 'test_loss'], loc='upper left')
plt.show()
plt.savefig('graphs/' + model_name + '.png')
