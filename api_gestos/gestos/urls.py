from django.urls import path
from.views import predecir_gesto 


urlpatterns = [
    path('', predecir_gesto, name='predecir_gesto')
]