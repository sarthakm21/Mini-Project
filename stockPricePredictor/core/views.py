import csv, io

# django imports
from django.shortcuts import render
from core.model import stackedLSTM, biDirectionalLSTM, classicLSTM

# Create your views here.

def getResults(request):

    fields = {}

    if request.method == "POST":
        data = request.FILES.get('excel').read().decode('UTF-8')
        model = request.POST.get('model')

        # open the file in the write mode
        with open('csv_file.csv', 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            io_string = io.StringIO(data)
            # next(io_string)
            for column in csv.reader(io_string, delimiter=',', quotechar="|"):
                writer.writerow(column)

        if model == "stacked":
            fields = stackedLSTM("csv_file.csv")
        elif model == "bidirectional":
            fields = biDirectionalLSTM("csv_file.csv")
        else:
            fields = classicLSTM("csv_file.csv")

    return render(request, 'results.html', fields)
