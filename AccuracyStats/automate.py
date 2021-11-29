import itertools
from funcs import stackedLSTM, biDirectionalLSTM, classicLSTM
import json

features = ['Open', 'High', 'Low', 'Close', 'Volume']
stockName = "CIPLA"
result = {}

def featuresetToString(featureset):
    return ', '.join(featureset)

for L in range(0, len(features)+1):
    for subset in itertools.combinations(features, L):
        featureset = list(subset)
        if len(featureset)<=1:
            continue
        print(featureset)
        with open(f'../dataset/{stockName}.csv', mode='r') as csv_file:
            res = stackedLSTM(csv_file, featureset)
            result[featuresetToString(featureset)] = res['r2_test']
        with open(f'{stockName}classicResults.json', 'w+') as f:
            json.dump(result, f, indent=4)

for L in range(0, len(features)+1):
    for subset in itertools.combinations(features, L):
        featureset = list(subset)
        if len(featureset)<=1:
            continue
        print(featureset)
        with open(f'../dataset/{stockName}.csv', mode='r') as csv_file:
            res = biDirectionalLSTM(csv_file, featureset)
            result[featuresetToString(featureset)] = res['r2_test']
        with open(f'{stockName}bidirectionalResults.json', 'w+') as f:
            json.dump(result, f, indent=4)

for L in range(0, len(features)+1):
    for subset in itertools.combinations(features, L):
        featureset = list(subset)
        if len(featureset)<=1:
            continue
        print(featureset)
        with open(f'../dataset/{stockName}.csv', mode='r') as csv_file:
            res = stackedLSTM(csv_file, featureset)
            result[featuresetToString(featureset)] = res['r2_test']
        with open(f'{stockName}stackedResults.json', 'w+') as f:
            json.dump(result, f, indent=4)
        
