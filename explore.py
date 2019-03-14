import keras
from dotenv import load_dotenv
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.utils.np_utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
import kappa
import matplotlib.pyplot as plt
import os
import pandas as pd
import pets
import tensorflow as tf
import time
import numpy as np
load_dotenv()

datapath = os.environ['DATA_PATH']

X = pets.load_input(datapath + '/train.csv', datapath +
                    '/train_sentiment/', keepPetId=False)
Y = pd.read_csv(datapath + '/train.csv', usecols=['AdoptionSpeed'])

# Y = pets.normalize_output(Y)
Y = to_categorical(Y)


input_units = X.shape[1]

model = pets.get_model(input_size=input_units)

model_name = time.strftime('%Y-%m-%d-%H-%M-%S')

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['mean_squared_error', 'accuracy'])

csvLogger = CSVLogger('data/' + model_name + '.csv')
checkpoint = ModelCheckpoint('models/' + model_name + '.h5',
                             monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')

callbacks = [csvLogger, checkpoint]
history = model.fit(X, Y, epochs=100, shuffle=True, batch_size=512,
                    validation_split=0.05, verbose=1, callbacks=callbacks)

scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# plt.subplot(121)
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model acc')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train_acc', 'test_acc'], loc='upper left')

# # summarize history for loss
# plt.subplot(122)
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train_loss', 'test_loss'], loc='upper left')
# plt.show()
# plt.savefig('graphs/' + model_name + '.png')
