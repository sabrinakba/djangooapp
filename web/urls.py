from django.urls import path
from django.contrib import admin
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='index'),
     path('add', views.add, name='add'),
    


    path('load-courses/', views.load_courses, name='ajax_load_courses'), # AJAX
]