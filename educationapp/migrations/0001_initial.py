# Generated by Django 4.1.3 on 2022-11-24 23:07

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="Person",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("edad", models.DecimalField(decimal_places=0, max_digits=3)),
                ("sexo", models.IntegerField()),
                ("estudios", models.IntegerField()),
                ("estudios_tutores", models.IntegerField()),
                (
                    "inversion_educacion",
                    models.DecimalField(decimal_places=2, max_digits=5),
                ),
                ("trabajo_relacionado", models.IntegerField()),
                ("salario", models.DecimalField(decimal_places=2, max_digits=5)),
                ("hijos", models.IntegerField()),
                ("institucion", models.IntegerField()),
                ("promedio", models.DecimalField(decimal_places=2, max_digits=5)),
                ("dificultad_aprendizaje", models.IntegerField()),
                ("retirado", models.IntegerField()),
                ("dificultad_trabajo", models.IntegerField()),
                ("repeticion_materia", models.IntegerField()),
                ("estudios_hijos", models.IntegerField()),
                ("latitude", models.DecimalField(decimal_places=6, max_digits=10)),
                ("longitude", models.DecimalField(decimal_places=6, max_digits=10)),
            ],
        ),
    ]
