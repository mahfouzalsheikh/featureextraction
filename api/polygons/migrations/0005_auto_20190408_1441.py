# Generated by Django 2.2 on 2019-04-08 14:41

import django.contrib.gis.db.models.fields
from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('polygons', '0004_auto_20190408_1436'),
    ]

    operations = [
        migrations.AlterField(
            model_name='boundery',
            name='polygon',
            field=django.contrib.gis.db.models.fields.PolygonField(blank=True, null=True, srid=4326),
        ),
    ]
