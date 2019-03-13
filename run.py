from dotenv import load_dotenv
import keras
import os
import pets

load_dotenv()

datapath = os.environ['DATA_PATH']

X = pets.load_input(datapath + '/test.csv', datapath +
                    '/test_sentiment/', keepPetId=True)
train_X = pets.load_input(datapath + '/train.csv',
                          datapath + '/train_sentiment/', keepPetId=True)

for test_column in X.columns:
    if test_column not in train_X.columns:
        X = X.drop([test_column], axis=1)

for train_column in train_X.columns:
    if train_column not in X.columns:
        # print('need to add ', train_column)
        X.insert(0, column=train_column, value=0)


X = X.sort_index(axis=1)

test_X = X
test_X = test_X.drop(columns=['PetID'], axis=1)

input_units = test_X.shape[1]
model = pets.get_model(input_size=input_units)
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.load_weights('models/2019-03-12-18-13-11.h5')

prediction = model.predict(test_X)

# for index, pred in prediction.iterrows():
#     print(index, pred)
# X['prediction'] = prediction

output = X[['PetID']]
output['AdoptionSpeed'] = pets.denormalize_output(prediction)
print(output)
# output[]
# print(prediction)
# print(pets.denormalize_output(prediction))

output.to_csv('submission.csv', index=False)
