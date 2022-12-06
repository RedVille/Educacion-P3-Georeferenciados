from django.db import models

# Create your models here.
class Person(models.Model):
    edad = models.DecimalField(max_digits=10, decimal_places=1)
    sexo = models.IntegerField()
    estudios = models.IntegerField()
    estudios_tutores = models.IntegerField()
    inversion_educacion = models.DecimalField(max_digits=10, decimal_places=1)
    trabajo_relacionado = models.IntegerField()
    salario = models.DecimalField(max_digits=10, decimal_places=2)
    hijos = models.IntegerField()
    institucion = models.IntegerField()
    promedio = models.DecimalField(max_digits=10, decimal_places=2)
    dificultad_aprendizaje = models.IntegerField()
    retirado = models.IntegerField()
    dificultad_trabajo = models.IntegerField()
    repeticion_materia = models.IntegerField()
    independiente = models.IntegerField()
    latitude = models.DecimalField(max_digits=10, decimal_places=6)
    longitude = models.DecimalField(max_digits=10, decimal_places=6)

    objects = models.Manager() # The default manager.