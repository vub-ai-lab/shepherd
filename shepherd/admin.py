from django.contrib import admin
from django.forms import ModelChoiceField
from .models import *

def only_own(s, request, **kwargs):
    qs = s.get_queryset(request)

    if request.user.is_superuser:
        return qs

    return qs.filter(**kwargs)

class AgentAdmin(admin.ModelAdmin):
    readonly_fields = ['learning_curve', 'latest_zip']

    # Normal users only see their own agents
    def get_queryset(self, request):
        return only_own(super(), request, owner=request.user)

class ParameterValueAdmin(admin.ModelAdmin):
    # Normal users can only change the parameters of their own agents
    def formfield_for_foreignkey(self, db_field, request, **kwargs):
        if db_field.name == 'agent' and not request.user.is_superuser:
            return ModelChoiceField(queryset=Agent.objects.filter(owner=request.user))

        return super().formfield_for_foreignkey(db_field, request, **kwargs)

    # Normal users only see their own agents
    def get_queryset(self, request):
        return only_own(super(), request, agent__owner=request.user)

class APIKeyAdmin(admin.ModelAdmin):
    # Normal users only see their own agents
    def get_queryset(self, request):
        return only_own(super(), request, agent__owner=request.user)

admin.site.register(Algorithm)
admin.site.register(Agent, AgentAdmin)
admin.site.register(APIKey, APIKeyAdmin)
admin.site.register(EpisodeReturn)
admin.site.register(Parameter)
admin.site.register(ParameterValue, ParameterValueAdmin)
