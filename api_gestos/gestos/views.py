from django.http import JsonResponse, HttpResponse
from tensorflow.keras.models import load_model
from django.views.decorators.csrf import csrf_exempt
import os
import cv2
import mediapipe as mp
import numpy as np

# Create your views here.

ruta_modelos = '../Modelos/106'

modelos = {}

# Cargar modelos
for i in os.listdir(ruta_modelos):
    if i.endswith('final.h5'):
        nombre_modelo = i.split('_')[1]
        modelos[nombre_modelo] = load_model(f"{ruta_modelos}/{i}")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

@csrf_exempt
def predecir_gesto(request):
    if request.method == 'POST':
        archivo = request.FILES.get('img')
        if archivo is None:
            return JsonResponse({'error': 'Imagen no proporcionada'}, status=400)

        file_bytes = archivo.read()
        np_arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        mano = recortar_mano(img)
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


def recortar_mano(image, margen=0.2):
    """Detecta la mano más alta en una imagen y devuelve un recorte cuadrado con margen extra."""
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = hands.process(image_rgb)
    if not results.multi_hand_landmarks:
        return None  # No se detectó ninguna mano

    min_y = float('inf')
    best_hand_bbox = None
    image_h, image_w, _ = image.shape

    for hand_landmarks in results.multi_hand_landmarks:
        x_coords = [int(landmark.x * image_w) for landmark in hand_landmarks.landmark]
        y_coords = [int(landmark.y * image_h) for landmark in hand_landmarks.landmark]

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        if y_min < min_y:
            min_y = y_min
            best_hand_bbox = (x_min, y_min, x_max, y_max)

    if best_hand_bbox is None:
        return None

    x_min, y_min, x_max, y_max = best_hand_bbox

    # Calcular el tamaño del cuadrado y agregar margen
    side_length = max(x_max - x_min, y_max - y_min)
    extra_space = int(side_length * margen)
    side_length += extra_space * 2  # Aumentamos por ambos lados

    # Centro del cuadrado
    center_x, center_y = (x_min + x_max) // 2, (y_min + y_max) // 2

    # Calcular nuevas coordenadas con margen extra
    x_min_sq = max(center_x - side_length // 2, 0)
    y_min_sq = max(center_y - side_length // 2, 0)
    x_max_sq = min(center_x + side_length // 2, image_w)
    y_max_sq = min(center_y + side_length // 2, image_h)

    return image[y_min_sq:y_max_sq, x_min_sq:x_max_sq]

def index(request):
    return HttpResponse("<h1>Hola mundo</h1>")