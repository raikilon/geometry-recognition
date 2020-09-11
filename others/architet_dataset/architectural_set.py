import json
import pandas as pd
import numpy as np
import os
import shutil

df = pd.read_csv("all.csv", header=0)

# get ShapeNetModels given taxonomy

# gets model id per category
dict_files = {}
with open('taxonomy.json') as json_file:
    data = json.load(json_file)
    for d in data:
        if len(d['children']):
            for c in d['children']:
                dict_files[c['name']] = {'id': d['synsetId'],
                                         'data': df.loc[df['subSynsetId'] == int(c['synsetId'])].modelId.to_numpy()}
        else:
            dict_files[d['name']] = {'id': d['synsetId'],
                                     'data': df.loc[df['synsetId'] == int(d['synsetId'])].modelId.to_numpy()}
    # print(data)
# print(files)
names = {}
names = dict.fromkeys(dict_files, 0)
print(names)
for subdir, dirs, files in os.walk("/Users/raikilon/Documents/Thesis/datasets/ShapeNetCore.v2"):
    for file in files:
        if file != "model_normalized.obj":
            continue
        check = False
        name = None
        # check if id is in list
        for key, value in dict_files.items():
            if value['id'] == subdir.split("/")[-3]:
                if subdir.split("/")[-2] in dict_files[key]['data']:
                    check = True
                    name = key
                    break
        if check:
            if not os.path.exists(
                    os.path.join("/Users/raikilon/Documents/Thesis/datasets/shapenetsubset", name, "train")):
                os.makedirs(os.path.join("/Users/raikilon/Documents/Thesis/datasets/shapenetsubset", name, "train"))
            shutil.copyfile(os.path.join(subdir, file),
                            os.path.join("/Users/raikilon/Documents/Thesis/datasets/shapenetsubset", name, "train",
                                         "{}_{}.obj".format(name, str(names[name]))))
            names[name] += 1



print("end")
