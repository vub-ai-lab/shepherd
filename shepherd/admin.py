# This file is part of Shepherd.
#
# Shepherd is free software: you can redistribute it and/or modify it under the
# terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# Shepherd is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with Shepherd. If not, see <https://www.gnu.org/licenses/>.

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
    readonly_fields = ['learning_curve', 'special_actions', 'creation_time', 'last_activity_time']
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
