import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from django.shortcuts import render

# scikit learn
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn import metrics



# Create your views here.
def index(request):
    return render(request, 'index.html')

def clusters(request):
    return render(request, "clusters.html")

def svm(request):
    return render(request, "svm.html")

def modeloSVM(request):

    # Cargar base de
    bc = datasets.load_breast_cancer()
    X = bc.data
    y = bc.target
    
    # Crear conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    #plt.scatter(X[:,0],X[:,1])
    print(X[0:3,0],y[0])

    # Normalizar datos mediante z-score
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    # Crear modelo de Support Vector Machine
    svc = SVC(C=1.0, random_state=1, kernel='linear')
    
    # Entrenar modelo
    svc.fit(X_train_std, y_train)

    # Prueba del modelo
    y_predict = svc.predict(X_train_std)
    
    # Calcular rendimiento
    
    # print("Porcentaje de clasificación %.3f" %metrics.accuracy_score(y_test, y_predict))
    print("Porcentaje de clasificación %.3f" %metrics.accuracy_score(y_train, y_predict))

    return render(request, "svm.html")