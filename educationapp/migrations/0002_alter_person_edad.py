# Generated by Django 4.1.3 on 2022-11-24 23:11

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("educationapp", "0001_initial"),
    ]

    operations = [
        migrations.AlterField(
            model_name="person",
            name="edad",
            field=models.DecimalField(decimal_places=2, max_digits=3),
        ),
    ]