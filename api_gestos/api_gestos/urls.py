from django.contrib import admin
from django.urls import path, include
from gestos import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='inicio'),
    path('predecir/gesto/', include('gestos.urls'))
]
