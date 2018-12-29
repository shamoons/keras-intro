from keras.models import Sequential
from keras.layers import Dense
import numpy
import pandas as pd

X = pd.read_csv(
    "data/train.csv", header=0, usecols=['Type', 'Age', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3', 'MaturitySize',	'FurLength',	'Vaccinated',	'Dewormed',	'Sterilized',	'Health',	'Quantity',	'Fee', 'VideoAmt', 'PhotoAmt', 'State'])
Y = pd.read_csv(
    "data/train.csv", header=0, usecols=['AdoptionSpeed'])

X = pd.get_dummies(X, columns=["Type", "Breed1",
                               "Breed2", "Color1", "Color2", "Color3", "Gender", "MaturitySize", "FurLength", "State"])
print(X)

Y = Y['AdoptionSpeed'].apply(lambda v: v / 4)

input_units = X.shape[1]

model = Sequential()
model.add(Dense(input_units, input_dim=input_units, activation='relu'))
model.add(Dense(input_units, activation='relu'))
model.add(Dense(input_units, activation='relu'))
model.add(Dense(input_units, activation='relu'))
model.add(Dense(input_units, activation='relu'))
model.add(Dense(input_units, activation='relu'))
model.add(Dense(input_units, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=250, batch_size=500,
          shuffle=True, validation_split=0.1, verbose=2)
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
