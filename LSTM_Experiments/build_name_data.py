__author__ = 'patrick'

import json
import os
import itertools as it

def get_gender_name(string):
    row = string.split(",")
    if len(row) > 1:
        gender = row[1]
        name = row[0]
        return [gender, name]

all_names = {'F': set(), 'M': set()}

for root, dirs, files in os.walk("names"):
    for name in files:
        if ".txt" in name:
            year_names = {}
            with open(os.path.join(root, name)) as file:
                text = file.read().split('\r\n')
                reduced_text = [get_gender_name(x) for x in text if x]
                # Collapsing the list of gender name pairs into dictionary for given year
                year_names = {k:tuple(x[1] for x in v) for k,v in it.groupby(sorted(reduced_text), key=lambda x: x[0])}
                for gender in year_names:
                    all_names[gender] = all_names[gender].union(set(year_names[gender]))

for gender in all_names:
    all_names[gender] = list(all_names[gender])

with open('babynames.json', 'w') as outfile:
    json.dump(all_names, outfile)