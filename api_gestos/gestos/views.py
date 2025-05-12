from django.http import JsonResponse, HttpResponse
from tensorflow.keras.models import load_model
from django.views.decorators.csrf import csrf_exempt
import os
import cv2
import numpy as np

# Create your views here.

ruta_modelos = '../Modelos/106'

modelos = {}

# Cargar modelos
for i in os.listdir(ruta_modelos):
    if i.endswith('final.h5'):
        nombre_modelo = i.split('_')[1]
        modelos[nombre_modelo] = load_model(f"{ruta_modelos}/{i}")


@csrf_exempt
def predecir_gesto(request):
    if request.method == 'POST':
        archivo = request.FILES.get('img')
        if archivo is None:
            return JsonResponse({'error': 'Imagen no proporcionada'}, status=400)

        file_bytes = archivo.read()
        np_arr = np.frombuffer(file_bytes, np.uint8)
        mano = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if mano is not None:
            mano_gray = cv2.cvtColor(mano, cv2.COLOR_BGR2GRAY)
            mano_resized = cv2.resize(mano_gray, (80, 80))
            mano_normalized = mano_resized / 255.0
            input_tensor = mano_normalized.reshape(1, 80, 80, 1)

            gesto_detectado = "NINGUNO"
            for gesto, modelo in modelos.items():
                prediccion = modelo.predict(input_tensor)[0][0]
                if prediccion > 0.5:
                    gesto_detectado = gesto.upper()
                    break
            
            return JsonResponse({'gesto': gesto_detectado}, status=200)
        else:
            return JsonResponse({'gesto': 'nh'}, status=200)


        

    else:
        return JsonResponse({'error': 'Solo se permite el método POST '}, status=405)


def index(request):
    return HttpResponse("<h1>Hola mundo</h1>")