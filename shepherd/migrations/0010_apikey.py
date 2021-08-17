# Generated by Django 3.2.5 on 2021-08-16 11:42

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('shepherd', '0009_episodereturn'),
    ]

    operations = [
        migrations.CreateModel(
            name='APIKey',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('key', models.CharField(default='2e5190db-ff33-4ad9-9693-55b310559907', max_length=36, unique=True, verbose_name='API Key string')),
                ('agent', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='shepherd.agent', verbose_name='Agent identified by this API key')),
            ],
        ),
    ]
