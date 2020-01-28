import os
import json
import matplotlib.pyplot as plt

path = "D:/Resilio Sync/dfv2/ShaneSmiskol-TOYOTA COROLLA 2017/df-data.29"

if path == "":
    path = input("Paste in file name and path of old format file: ")

with open(path, 'r') as f:
    data = f.read().replace("'", '"').replace('False', 'false').replace('True', 'true').split('\n')

data_parsed = []
for sample in data:
    try:
        data_parsed.append(json.loads(sample))
    except:
        pass

if not isinstance(data_parsed[0], dict):
    raise Exception("Already new format!")

keys = list(data_parsed[0].keys())

print("Sorting values...", flush=True)
new_format = [keys]
for sample in data_parsed:
    values_sorted = [sample[key] for key in keys]
    new_format.append(values_sorted)
print("Writing!", flush=True)
with open(path + ".new", "w") as f:
    f.write('{}\n'.format('\n'.join(map(str, new_format))))