# package imports
from keras.layers import LSTM, Dense, Dropout
from pyexcel_xlsx import get_data as xlsx_get
from sklearn import linear_model
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# django imports
from django.shortcuts import render

# Create your views here.


def getResults(request):

    fields = {}

    if request.method == "POST":
        data = request.FILES.get('excel')
        data = xlsx_get(data)
        model = request.POST.get('model')
        print(data, model, request.POST)

    return render(request, 'results.html', fields)
