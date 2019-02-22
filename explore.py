from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
import pandas as pd
from tensorflow.python.client import device_lib
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import kappa
import os
import time
load_dotenv()

datapath = os.environ['DATA_PATH']

X = pd.read_csv(datapath + '/train.csv')
Y = pd.read_csv(datapath + '/train.csv', header=0, usecols=['AdoptionSpeed'])
Y = pd.get_dummies(Y['AdoptionSpeed'], columns=['AdoptionSpeed'])
X = X.drop(['Description', 'AdoptionSpeed', 'Name', 'PetID'], axis=1)

X = pd.get_dummies(X, columns=['Type', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3', 'MaturitySize',
                               'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'State', 'RescuerID'])

X['Age'] = X['Age'] / X['Age'].max()
X['Quantity'] = X['Quantity'] / X['Quantity'].max()
X['Fee'] = X['Fee'] / X['Fee'].max()
X['VideoAmt'] = X['VideoAmt'] / X['VideoAmt'].max()
X['PhotoAmt'] = X['PhotoAmt'] / X['PhotoAmt'].max()

input_units = X.shape[1]
output_units = Y.shape[1]

model = Sequential()
model.add(Dense(input_units, input_dim=input_units, activation='relu'))

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

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

history = model.fit(X, Y, epochs=20, shuffle=True, batch_size=500,
                    validation_split=0.05, verbose=1)

scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

model_name = time.strftime('%Y-%m-%d-%H-%M-%S')
model.save('models/' + model_name + '.h5')

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('graphs/val' + model_name + '.png')
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('graphs/acc' + model_name + '.png')
