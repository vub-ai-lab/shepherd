# Generated by Django 3.2.4 on 2021-06-14 13:20

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('shepherd', '0006_auto_20210614_1317'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='parameter',
            name='algo',
        ),
        migrations.AddField(
            model_name='parameter',
            name='algo',
            field=models.ForeignKey(default=1, on_delete=django.db.models.deletion.CASCADE, to='shepherd.algorithm', verbose_name='Algorithms that has this parameter'),
            preserve_default=False,
        ),
    ]
