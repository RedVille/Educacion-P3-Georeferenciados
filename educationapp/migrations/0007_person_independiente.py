# Generated by Django 3.2.5 on 2022-12-06 06:26

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('educationapp', '0006_auto_20221206_0022'),
    ]

    operations = [
        migrations.AddField(
            model_name='person',
            name='independiente',
            field=models.IntegerField(default=1),
            preserve_default=False,
        ),
    ]
