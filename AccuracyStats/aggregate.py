import json
import math

classicResult = {}
stackedResult = {}
bidirectionalResult = {}
stocks = ['DRREDDY', 'HDFCBANK', 'RELIANCE', 'SBIN', 'TATASTEEL']

for stock in stocks:
    # Read the data from the file
    with open(stock + 'classicResults.json', 'r') as f:
        stClRes = json.load(f)
        for key in stClRes:
            stClRes[key] = stClRes[key] ** 2
        if classicResult == {}:
            classicResult = stClRes
        else:
            for key in stClRes:
                classicResult[key] += stClRes[key]

    with open(stock + 'stackedResults.json', 'r') as f:
        stStRes = json.load(f)
        for key in stStRes:
            stStRes[key] = stStRes[key] ** 2
        if stackedResult == {}:
            stackedResult = stStRes
        else:
            for key in stStRes:
                stackedResult[key] += stStRes[key]

    with open(stock + 'bidirectionalResults.json', 'r') as f:
        stBdRes = json.load(f)
        for key in stBdRes:
            stBdRes[key] = stBdRes[key] ** 2
        if bidirectionalResult == {}:
            bidirectionalResult = stBdRes
        else:
            for key in stBdRes:
                bidirectionalResult[key] += stBdRes[key]


for key in classicResult:
    classicResult[key] = classicResult[key] / len(stocks)
    classicResult[key] = math.sqrt(classicResult[key])

for key in stackedResult:
    stackedResult[key] = stackedResult[key] / len(stocks)
    stackedResult[key] = math.sqrt(stackedResult[key])

for key in bidirectionalResult:
    bidirectionalResult[key] = bidirectionalResult[key] / len(stocks)
    bidirectionalResult[key] = math.sqrt(bidirectionalResult[key])

# Write the data to the file
with open('ZZFINALclassicResults.json', 'w') as f:
    json.dump(classicResult, f, indent=4)

with open('ZZFINALstackedResults.json', 'w') as f:
    json.dump(stackedResult, f, indent=4)

with open('ZZFINALbidirectionalResults.json', 'w') as f:
    json.dump(bidirectionalResult, f, indent=4)


        