import numpy as np
import pandas as pd
from django.shortcuts import render
from educationapp.models import Person
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import io
import urllib, base64
from django.views.decorators.csrf import csrf_exempt

# scikit learn
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn import metrics

# Extraer datos de información de la base de datos
def get_data(request):
    file = open("static/data.csv", "r")

    df = pd.read_csv(file)

    # Delete rows that include 50+ in age
    df = df[df['Ingrese su edad'].str.contains('50+')==False]

    # Get the age from the head of the column
    dfHead = df.head(14)
    age = df['Ingrese su edad'].head(14).str.split("-", n = 1, expand = True)

    # Drop the head of the df
    dfTail = df.iloc[14: , :]

    # Rename columns
    age.columns = ['A', 'B']

    # Parse into int
    age[['A', 'B']] = age[['A', 'B']].astype(int)

    # Get the mean from two columns
    age['Ingrese su edad'] = age[['A','B']].mean(axis=1)

    # Assign the age
    dfHead['Ingrese su edad'] = age['Ingrese su edad']

    # Append
    df = dfHead
    df = df.append(dfTail)

    # Clean the anual investment in education
    current = '¿Cuánto invierte en educación al año?'
    df.loc[df[current].str.contains('<10K', case=False), current] = '5000'
    df.loc[df[current].str.contains('10k-25k', case=False), current] = '17500'
    df.loc[df[current].str.contains('75k-100k', case=False), current] = '87500'
    df.loc[df[current].str.contains('100k+', case=False), current] = '120000'

    # Parse into float
    df[current] = df[current].astype(float)

    # Replace the 0 for a 1
    df[current] = df[current].replace(0,1)

    current = '¿Dentro de qué rango salarial te encuentras actualmente?'
    df.loc[df[current].str.contains('<10K', case=False), current] = '5000'
    df.loc[df[current].str.contains('10k - 25k', case=False), current] = '17500'
    df.loc[df[current].str.contains('25k - 50k', case=False), current] = '37500'

    # Parse into float
    df[current] = df[current].astype(float)

    # Replace the 0 for a 1
    df[current] = df[current].replace(0,1)

    # Clean the anual income
    current = '¿Cuál era tu promedio de calificaciones?'
    df.loc[df[current].str.contains('8 - 9', case=False), current] = '8.5'

    # Parse into float
    df[current] = df[current].astype(float)

    # Replace the 0 for a 1
    df[current] = df[current].replace(0,1)

    columns = ['En caso de que haber estudiado una carrera universitaria o estar en proceso de, ¿su trabajo actual está relacionado con ella?',
           '¿En algún momento interrumpió sus estudios por su situación económica?',
           '¿En algún momento interrumpió sus estudios por su situación de salud?',
           '¿En algún momento interrumpió sus estudios por su situación familiar?',
           '¿Es independiente económicamente?',
           '¿Tienes alguna discapacidad?',
           '¿Cambias o cambiabas constantemente tu ubicación geográfica?',
           '¿Sufrías o sufres de acoso escolar?',
           '¿Tienes o tuviste dificultades para aprender algo nuevo?',
           '¿Se encuentra o se encontraba muy retirado tu centros de estudios de tu hogar?',
           '¿Has tenido alguna dificultad para conseguir trabajo?',
           '¿Repetiste alguna materia en la escuela?',
           '¿Repetiste algún año de estudios?']

    for val in columns:
        # We define the current column
        current = val
        # All values that say yes are replaced by a 1
        df.loc[df[current].str.contains('Si', case=False), current] = '1'
        # All values that say no are replaced by a -1
        df.loc[df[current].str.contains('No', case=False), current] = '-1'

    # We define the current column
    current = 'Ingrese su género'
    # All values that say yes are replaced by a 1
    df.loc[df[current].str.contains('Masculino', case=False), current] = '1'
    # All values that say no are replaced by a -1
    df.loc[df[current].str.contains('Femenino', case=False), current] = '2'

    columns = ['¿Cuál es su grado máximo de estudios?',
           '¿Cuál es el grado máximo de estudios de sus padres/tutores?']

    for val in columns:
        # We define the current column
        current = val
        # We replace the top level of study
        df.loc[df[current].str.contains('Nivel básico', case=False), current] = '1'
        df.loc[df[current].str.contains('Nivel medio superior', case=False), current] = '2'
        df.loc[df[current].str.contains('Nivel superior', case=False), current] = '3'
        df.loc[df[current].str.contains('Posgrado', case=False), current] = '4'

    columns = ['¿En qué tipo de institución realizó sus estudios de nivel básico?',
          '¿En qué tipo de institución realizó sus estudios de nivel medio-superior?',
          '¿En qué tipo de institución realizó sus estudios de nivel superior?',
          '¿En qué tipo de institución realizó sus estudios de posgrado?']

    for val in columns:
        # We define the current column
        current = val
        # We replace where the person had their studies
        df.loc[df[current].str.contains('No realizó', case=False), current] = '1'
        df.loc[df[current].str.contains('Pública', case=False), current] = '2'
        df.loc[df[current].str.contains('Ambas', case=False), current] = '3'
        df.loc[df[current].str.contains('Privada', case=False), current] = '4'

    # We add the city, the state and the country for the library to find the address easier
    current = '¿En qué colonia de León vives?'
    df[current] =  df[current].astype(str) + ', León, Guanajuato, México'

    #Importing the Nominatim geocoder class 
    from geopy.geocoders import Nominatim
    from geopy.extra.rate_limiter import RateLimiter

    current = '¿En qué colonia de León vives?'

    #Creating an instance of Nominatim Class
    geolocator = Nominatim(user_agent="my_request")
    
    #applying the rate limiter wrapper
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    
    #Applying the method to pandas DataFrame
    df['location'] = df[current].apply(geocode)
    df['Lat'] = df['location'].apply(lambda x: x.latitude if x else None)
    df['Lon'] = df['location'].apply(lambda x: x.longitude if x else None)

    df.dropna(inplace = True)
    
    for index, row in df.iterrows():
        print(row)
        person = Person()
        person.edad = row['Ingrese su edad']
        person.sexo = row['Ingrese su género']
        person.estudios = row['¿Cuál es su grado máximo de estudios?']
        person.estudios_tutores = row['¿Cuál es el grado máximo de estudios de sus padres/tutores?']
        person.inversion_educacion = row['¿Cuánto invierte en educación al año?']
        person.trabajo_relacionado = row['En caso de que haber estudiado una carrera universitaria o estar en proceso de, ¿su trabajo actual está relacionado con ella?']
        person.salario = row['¿Dentro de qué rango salarial te encuentras actualmente?']
        person.hijos = row['¿Cuántos hijos tienes?']
        person.institucion = row['¿En qué tipo de institución realizó sus estudios de nivel básico?']
        person.promedio = row["¿Cuál era tu promedio de calificaciones?"]
        person.dificultad_aprendizaje = row['¿Tienes o tuviste dificultades para aprender algo nuevo?']
        person.retirado = row['¿Se encuentra o se encontraba muy retirado tu centros de estudios de tu hogar?']
        person.dificultad_trabajo = row['¿Has tenido alguna dificultad para conseguir trabajo?']
        person.repeticion_materia = row['¿Repetiste alguna materia en la escuela?']
        person.independiente = row['¿Es independiente económicamente?']
        person.latitude = row['Lat']
        person.longitude = row['Lon']
        print(person)
        person.save()

    return render(request, "svm.html")

# Create your views here.
def index(request):
    return render(request, 'index.html')

@csrf_exempt
def clusters(request):

    # Usamos el metodo elbow para determinar la cantidad de clusters
    persons = Person.objects.all().values()
    df = pd.DataFrame(persons)

    X = df.iloc[:, [15, 16]].values

    # Usando Elbow para encontrar los clusters óptimos
    # Se determina el Valor WCSS
    wcss = []
    for i in range(1, 10):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 45)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)  # Parámetro Inercia

    print(wcss)
    plt.figure(figsize=(10,5))
    sns.lineplot(range(1, 10), wcss,marker='o',color='blue')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri = urllib.parse.quote(string)

    markers_icons = [
        "http://maps.google.com/mapfiles/ms/icons/blue-dot.png",
        "http://maps.google.com/mapfiles/ms/icons/red-dot.png",
        "http://maps.google.com/mapfiles/ms/icons/green-dot.png",
        "http://maps.google.com/mapfiles/ms/icons/yellow-dot.png",
        "http://maps.google.com/mapfiles/ms/icons/purple-dot.png",
        "http://maps.google.com/mapfiles/ms/icons/pink-dot.png",
        "http://maps.google.com/mapfiles/ms/icons/orange-dot.png",
        "http://maps.google.com/mapfiles/ms/icons/ltblue-dot.png",
        "http://maps.google.com/mapfiles/ms/icons/ltgreen-dot.png"
    ]

    if request.method == 'POST':
        k = int(request.POST.get('cluster'))
        kmeans = KMeans(n_clusters = k, init = 'k-means++', random_state = 45)
        y_kmeans = kmeans.fit_predict(X)

        x = []
        for i in range(0, len(X)):
            x.append({
                'lat': X[i][0],
                'lon': X[i][1],
                "index": i
            })

        return render(request, 'clusters.html', {'k': k, 'data': uri, 'y_kmeans': y_kmeans.tolist(), 'X': x, 'markers_icons': markers_icons})

    return render(request, "clusters.html", {'data': uri, 'k': 0, 'y_kmeans': [], 'X': [], 'markers_icons': markers_icons})

def svm(request):
    return render(request, "svm.html")

@csrf_exempt
def modeloSVM(request):
    # Llamamos datos del modelo
    personas = Person.objects.all().values()
  #  dfPersonas = pd.DataFrame(personas)

    #Categorias
    #Lista completa con personas 
    #posgrado = dfPersonas[dfPersonas.estudios == 2]
    datosGenerales = pd.DataFrame(personas)

    if request.method == 'POST':

        markers_icons = [
        "http://maps.google.com/mapfiles/ms/icons/blue-dot.png",
        "http://maps.google.com/mapfiles/ms/icons/yellow-dot.png"
        ]

        #Agregar personas clasificadas
        #HOMBRES
        if request.POST.get('radiogenero') == "1":
           datosGenerales = datosGenerales[datosGenerales.sexo == 1]

        #MUJERES
        if request.POST.get('radiogenero') == "2":
           datosGenerales = datosGenerales[datosGenerales.sexo == 2]

        #BASICO PADRES
        if request.POST.get('radio') == "1":
            datosGenerales = datosGenerales[datosGenerales.estudios_tutores == 1]
        
        #SUPERIOR PADRES
        if request.POST.get('radio') == "2":
            datosGenerales = datosGenerales[datosGenerales.estudios_tutores != 1]

        #TIENE HIJOS
        if request.POST.get('hijos') == "1":
            datosGenerales = datosGenerales[datosGenerales.hijos != 0]

        #INDEPENDIENTE
        if request.POST.get('independiente') == "1":
            datosGenerales = datosGenerales[datosGenerales.independiente == 1]

        #DIFICULTAD CONSEGUIR TRABAJO
        if request.POST.get('dificultad') == "1":
           datosGenerales = datosGenerales[datosGenerales.dificultad_trabajo == 1]                                 

        personasPosgrado = datosGenerales[datosGenerales.estudios == 4]
        personasNoPosgrado = datosGenerales[datosGenerales.estudios != 4]

        return render(request, "svm.html",{ "personasPosgrado" : personasPosgrado, "personasNoPosgrado" : personasNoPosgrado})

    return render(request, "svm.html",{ "personasPosgrado" : [], "personasNoPosgrado" : []})

