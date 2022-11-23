from django.shortcuts import render

# Create your views here.
def index(request):
    return render(request, 'index.html')

def clusters(request):
    return render(request, "clusters.html")

def svm(request):
    return render(request, "svm.html")