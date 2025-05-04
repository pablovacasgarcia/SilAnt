import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import os
import requests
import threading
import functions

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

gesto_detectado = "NO DETECTADO"
gesto_anterior = None
menu_activo = False
mostrar_menu = False
frame_mostrar_menu = None

acciones_por_gesto = {
    "UP": lambda: functions.cambiar_volumen(0.2),
    "DOWN": lambda: functions.cambiar_volumen(-0.2),
    "STOP": functions.pausar_reproduccion,
    "PREV": functions.cancion_anterior,
    "NEXT": functions.cancion_siguiente,
}

def ejecutar_accion_por_gesto(gesto):
    global gesto_anterior, menu_activo, mostrar_menu

    if menu_activo:
        # Si el menú está activo, solo responde a gestos del menú
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

    if gesto != gesto_anterior:
        if gesto == "OPEN":
            # Activar menú
            print("Menú activado. Esperando selección...")
            menu_activo = True
            mostrar_menu = True
        else:
            accion = acciones_por_gesto.get(gesto)
            if accion:
                print(f"Ejecutando acción para gesto: {gesto}")
                accion()
        gesto_anterior = gesto

def enviar_imagen(frame):
    global gesto_detectado
    _, img_encoded = cv2.imencode('.jpg', frame)
    try:
        response = requests.post(
            'http://localhost:8000/predecir/gesto/',
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

def mostrar_menu_en_frame(frame):
    cv2.putText(frame, "Abrir aplicación:", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame, "1 -> YouTube", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, "2 -> Spotify", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, "3 -> Cerrar menu", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

def detectar_mirada_y_gestos():
    global mostrar_menu
    cap = cv2.VideoCapture(0)
    frame_counter = 0
    prediccion_intervalo = 50  

    # Ventana en pantalla completa
    cv2.namedWindow("Gesto + Mirada", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Gesto + Mirada", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_result = face_mesh.process(rgb_frame)

        mirando = False

        if face_result.multi_face_landmarks:
            for face_landmarks in face_result.multi_face_landmarks:
                left_eye = face_landmarks.landmark[159]
                right_eye = face_landmarks.landmark[386]
                nose = face_landmarks.landmark[1]
                dx = abs(nose.x - ((left_eye.x + right_eye.x) / 2))

                if dx < 0.02:
                    mirando = True
                    if frame_counter % prediccion_intervalo == 0:
                        frame_counter = 0
                        threading.Thread(target=enviar_imagen, args=(frame.copy(),)).start()

        frame_counter += 1

        if mostrar_menu:
            mostrar_menu_en_frame(frame)
            if not menu_activo:
                mostrar_menu = False  # Ocultar menú una vez se haya ejecutado algo

        # Mostrar círculo verde si está mirando a la cámara
        if mirando:
            alto, ancho, _ = frame.shape
            centro_x = ancho // 2
            centro_y = alto - 50
            cv2.circle(frame, (centro_x, centro_y), 10, (0, 255, 0), -1)

        
        cv2.imshow("Gesto + Mirada", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detectar_mirada_y_gestos()
