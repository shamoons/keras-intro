from keras.models import Sequential
from keras.layers import Dense, Dropout
import pandas as pd

datapath = '/Volumes/MoonFlash/kaggle-data'

X = pd.read_csv(datapath + '/train.csv')
Y = pd.read_csv(datapath + '/train.csv', header=0, usecols=['AdoptionSpeed'])
Y = pd.get_dummies(Y['AdoptionSpeed'], columns=['AdoptionSpeed'])
X = X.drop(['Description', 'AdoptionSpeed', 'Name', 'PetID'], axis=1)
X = pd.get_dummies(X, columns=['Type', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3', 'MaturitySize',
                               'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'State', 'RescuerID'])
X['Age'] = X['Age'].apply(lambda v: v / max(X['Age']))
X['Quantity'] = X['Quantity'].apply(lambda v: v / max(X['Quantity']))
X['VideoAmt'] = X['VideoAmt'].apply(lambda v: v / max(X['VideoAmt']))
X['PhotoAmt'] = X['PhotoAmt'].apply(lambda v: v / max(X['PhotoAmt']))

input_units = X.shape[1]
output_units = Y.shape[1]
print(output_units)
# model = Sequential()
# model.add(Dense(input_units, input_dim=input_units, activation='relu'))

# model.add(Dense(output_units, activation='softmax'))

# model.compile(loss='binary_crossentropy',
#               optimizer='adam', metrics=['accuracy'])
# model.fit(X, Y, epochs=250, batch_size=250,
#           shuffle=True, validation_split=0.05, verbose=2)

# scores = model.evaluate(X, Y)
# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# X = []

# Type
# pet_type = keras.utils.to_categorical(train['Type'], dtype='int32')
# X.append(pet_type)

# Age
# age = train['Age'].apply(lambda v: v / max(train['Age']))
# print(age)
# X.append(age)


# test = pd.read_csv(datapath + '/test/test.csv')
# sub = pd.read_csv(datapath + '/test/sample_submission.csv')

# train['dataset_type'] = 'train'
# test['dataset_type'] = 'test'
# all_data = pd.concat([train, test])
# print(train['Type'])
# print(X)
# print(Y)
