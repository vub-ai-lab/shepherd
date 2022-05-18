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

from django.urls import path

from . import views

urlpatterns = [
    path('login_user/', views.login_user, name='login_user'),
    path('env/', views.env, name='env'),
    path('send_curve/', views.send_curve, name='send_curve'),
    path('generate_zip/', views.generate_zip, name='generate_zip'),
    path('delete_zip/', views.delete_zip, name='delete_zip'),
    path('delete_curve/', views.delete_curve, name='delete_curve'),
    path('export_curve_CSV/', views.export_curve_CSV, name='export_curve_CSV'),
    path('kill_processes/', views.kill_processes, name='kill_processes'),
]
