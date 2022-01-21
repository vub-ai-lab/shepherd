from django.contrib import admin
from django.forms import ModelChoiceField
from .models import *

def only_own(s, request, **kwargs):
    qs = s.get_queryset(request)

    if request.user.is_superuser:
        return qs

    return qs.filter(**kwargs)

class ParameterValueInline(admin.TabularInline):
    model = ParameterValue
    extra = 1

    def formfield_for_foreignkey(self, db_field, request, **kwargs):
        if db_field.name == "param":
            # List per algorithm
            kwargs["queryset"] = Parameter.objects.order_by('algo')

        return super().formfield_for_foreignkey(db_field, request, **kwargs)

class AgentAdmin(admin.ModelAdmin):
    readonly_fields = ['learning_curve', 'latest_agent_model_zip_file', 'creation_time', 'last_activity_time']
    inlines = (ParameterValueInline,)

    # Normal users only see their own agents
    def get_queryset(self, request):
        return only_own(super(), request, owner=request.user)

class APIKeyAdmin(admin.ModelAdmin):
    # Normal users only see their own agents
    def get_queryset(self, request):
        return only_own(super(), request, agent__owner=request.user)

admin.site.register(Algorithm)
admin.site.register(Agent, AgentAdmin)
admin.site.register(APIKey, APIKeyAdmin)
admin.site.register(EpisodeReturn)
admin.site.register(Parameter)
