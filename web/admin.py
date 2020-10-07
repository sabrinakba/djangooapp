from django.contrib import admin

# Register your models here.
from web.models import Programming, Course

admin.site.register(Programming)
admin.site.register(Course)
