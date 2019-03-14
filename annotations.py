import json
import os
import pydash
from dotenv import load_dotenv
load_dotenv()

datapath = os.environ['DATA_PATH']

files = os.listdir(datapath + '/train_metadata')

annotatedObjects = {}

for file in files:
    metadata_file = datapath + '/train_metadata/' + file
    json_data = json.loads(open(metadata_file).read())

    labelAnnotations = []
    if 'labelAnnotations' in json_data:
        labelAnnotations = json_data['labelAnnotations']
        for annotation in labelAnnotations:
            if annotation["description"] in annotatedObjects:
                annotatedObjects[annotation["description"]] += 1
            else:
                annotatedObjects[annotation["description"]] = 1

finalObjects = {}
for obj in annotatedObjects:
    if annotatedObjects[obj] >= 50:
        finalObjects[obj] = annotatedObjects[obj]


with open('image-annotations.json', 'w') as outfile:
    json.dump(finalObjects, outfile)
