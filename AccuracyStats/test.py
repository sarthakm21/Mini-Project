import json

f1 = open('ZZFINALclassicResults.json', 'r')
c1 = json.load(f1)
f2 = open('ZZFINALstackedResults.json', 'r')
c2 = json.load(f2)
f3 = open('ZZFINALbidirectionalResults.json', 'r')
c3 = json.load(f3)

res = []
for key in c1:
    temp = {}
    temp['Features'] = key
    temp['Classic LSTM'] = round(1-c1[key], 5)
    temp['Stacked LSTM'] = round(1-c2[key], 5)
    temp['Bidirectional LSTM'] = round(1-c3[key], 5)
    res.append(temp)

with open('results.json', 'w') as f:
    json.dump(res, f, indent=4)
