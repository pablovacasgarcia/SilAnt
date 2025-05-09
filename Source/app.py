# Importación de librerías necesarias
import cv2  # OpenCV para procesamiento de imágenes
import mediapipe as mp  # MediaPipe para detección de rostros y puntos faciales
import numpy as np
from tensorflow.keras.models import load_model  # Para cargar modelos de IA (no usado aquí directamente)
import os
import requests  # Para hacer peticiones HTTP
import threading  # Para ejecutar funciones en segundo plano (multihilo)
import functions  # Módulo personalizado con funciones específicas (como controlar música o abrir apps)

# Inicialización del detector de malla facial de MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Variables globales para el estado del sistema
gesto_detectado = "NO DETECTADO"
gesto_anterior = None
menu_activo = False
mostrar_menu = False
frame_mostrar_menu = None  # (No usado directamente, pero reservado para mostrar menú en un frame)

# Diccionario que asocia gestos con funciones específicas
acciones_por_gesto = {
    "UP": lambda: functions.cambiar_volumen(0.2),
    "DOWN": lambda: functions.cambiar_volumen(-0.2),
    "STOP": functions.pausar_reproduccion,
    "PREV": functions.cancion_anterior,
    "NEXT": functions.cancion_siguiente,
}

# Ejecuta una acción según el gesto detectado
def ejecutar_accion_por_gesto(gesto):
    global gesto_anterior, menu_activo, mostrar_menu

    if menu_activo:
        # Si el menú está activo, responde solo a gestos de selección
        if gesto == "UP":
            print("Abriendo YouTube...")
            functions.abrir_youtube()
            menu_activo = False
        elif gesto == "OPEN":
            print("Abriendo Spotify...")
            functions.abrir_spotify()
            menu_activo = False
        elif gesto == "THREE":
            print("Cerrando menú.")
            menu_activo = False
        return

    # Si el gesto es nuevo (no es el mismo que el anterior)
    if gesto != gesto_anterior:
        if gesto == "OPEN":
            # Activar menú visual
            print("Menú activado. Esperando selección...")
            menu_activo = True
            mostrar_menu = True
        else:
            # Ejecutar acción definida para ese gesto
            accion = acciones_por_gesto.get(gesto)
            if accion:
                print(f"Ejecutando acción para gesto: {gesto}")
                accion()
        gesto_anterior = gesto

# Envía un frame al servidor local para predecir el gesto
def enviar_imagen(frame):
    global gesto_detectado
    _, img_encoded = cv2.imencode('.jpg', frame)  # Codifica el frame a JPEG
    try:
        response = requests.post(
            'http://localhost:8000/predecir/gesto/',  # Ruta del endpoint de predicción
            files={'img': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')}
        )
        if response.status_code == 200:
            data = response.json()
            gesto_detectado = data.get('gesto', 'NO DETECTADO')
            ejecutar_accion_por_gesto(gesto_detectado)
        else:
            print("Error:", response.status_code)
    except Exception as e:
        print("Excepción al enviar imagen:", e)

# Dibuja el menú sobre el frame cuando está activo
def mostrar_menu_en_frame(frame):
    cv2.putText(frame, "Abrir aplicacion:", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame, "1 -> YouTube", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, "2 -> Spotify", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, "3 -> Cerrar menu", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

# Función principal que detecta si el usuario está mirando a la cámara y reconoce gestos
def detectar_mirada_y_gestos():
    global mostrar_menu
    cap = cv2.VideoCapture(0)  # Captura de video desde la cámara
    frame_counter = 0
    prediccion_intervalo = 50  # Intervalo entre predicciones (cada 50 frames)

    # Ventana de visualización en modo pantalla completa
    cv2.namedWindow("Gesto + Mirada", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Gesto + Mirada", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Espejo para parecerse más a un reflejo
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Conversión a RGB para MediaPipe
        face_result = face_mesh.process(rgb_frame)

        mirando = False  # Indica si el usuario está mirando al centro

        if face_result.multi_face_landmarks:
            for face_landmarks in face_result.multi_face_landmarks:
                # Extrae puntos clave: ojos y nariz
                left_eye = face_landmarks.landmark[159]
                right_eye = face_landmarks.landmark[386]
                nose = face_landmarks.landmark[1]

                # Calcula la distancia horizontal entre nariz y el centro de los ojos
                dx = abs(nose.x - ((left_eye.x + right_eye.x) / 2))

                if dx < 0.02:
                    mirando = True  # Se considera que el usuario está mirando al frente

                    # Cada cierto número de frames, envía una imagen para predecir el gesto
                    if frame_counter % prediccion_intervalo == 0:
                        frame_counter = 0
                        threading.Thread(target=enviar_imagen, args=(frame.copy(),)).start()

        frame_counter += 1

        # Si se debe mostrar el menú, lo dibuja sobre el frame
        if mostrar_menu:
            mostrar_menu_en_frame(frame)
            if not menu_activo:
                mostrar_menu = False  # Oculta menú si ya no está activo

        # Dibuja un círculo verde si el usuario está mirando a la cámara
        if mirando:
            alto, ancho, _ = frame.shape
            centro_x = ancho // 2
            centro_y = alto - 50
            cv2.circle(frame, (centro_x, centro_y), 10, (0, 255, 0), -1)

        # Muestra el frame en pantalla
        cv2.imshow("Gesto + Mirada", frame)

        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libera recursos
    cap.release()
    cv2.destroyAllWindows()

# Punto de entrada principal
if __name__ == "__main__":
    detectar_mirada_y_gestos()
