from django.contrib import admin
from .models import *

class AgentAdmin(admin.ModelAdmin):
    readonly_fields = ['learning_curve']
    

admin.site.register(Algorithm)
admin.site.register(Agent, AgentAdmin)
admin.site.register(APIKey)
admin.site.register(EpisodeReturn)
admin.site.register(Parameter)
admin.site.register(ParameterValue)
