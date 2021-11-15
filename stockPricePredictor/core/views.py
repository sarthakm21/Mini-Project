# package imports
from pyexcel_xlsx import get_data as xlsx_get

# django imports
from django.shortcuts import render
from core.model import stackedLSTM, biDirectionalLSTM, classicLSTM

# Create your views here.


def getResults(request):

    fields = {}

    if request.method == "POST":
        data = request.FILES.get('excel')
        data = xlsx_get(data)
        model = request.POST.get('model')
        print(data, model, request.POST)

    return render(request, 'results.html', fields)
