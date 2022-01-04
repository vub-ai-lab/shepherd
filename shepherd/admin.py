from django.contrib import admin
from .models import *

class AgentAdmin(admin.ModelAdmin):
    readonly_fields = ['learning_curve', 'latest_zip']
    

admin.site.register(Algorithm)
admin.site.register(Agent, AgentAdmin)
admin.site.register(APIKey)
admin.site.register(EpisodeReturn)
admin.site.register(Parameter)
admin.site.register(ParameterValue)
