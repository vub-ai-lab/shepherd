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

from django.apps import AppConfig

class ShepherdConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'shepherd'
