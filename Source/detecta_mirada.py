import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import os

modelos = {}
dir = '102'
# Cargar modelos
for i in os.listdir(f'./Modelos/{dir}'):
    if i.endswith('final.h5'):
        nombre_modelo = i.split('_')[1]
        modelos[nombre_modelo] = load_model(f"./Modelos/{dir}/{i}")


# Inicializar MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

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

def detectar_mirada_y_gestos():
    cap = cv2.VideoCapture(0)
    frame_counter = 0
    prediccion_intervalo = 10

    gesto_detectado = "NO DETECTADO"

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_result = face_mesh.process(rgb_frame)

        if face_result.multi_face_landmarks:
            for face_landmarks in face_result.multi_face_landmarks:
                left_eye = face_landmarks.landmark[159]
                right_eye = face_landmarks.landmark[386]
                nose = face_landmarks.landmark[1]
                dx = abs(nose.x - ((left_eye.x + right_eye.x) / 2))

                if dx < 0.02:
                    cv2.putText(frame, "Mirando a la camara", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    if frame_counter % prediccion_intervalo == 0:
                        frame_counter = 0
                        mano = recortar_mano(frame)
                        if mano is not None:
                            mano_gray = cv2.cvtColor(mano, cv2.COLOR_BGR2GRAY)
                            mano_resized = cv2.resize(mano_gray, (48, 48))
                            mano_normalized = mano_resized / 255.0
                            input_tensor = mano_normalized.reshape(1, 48, 48, 1)

                            gesto_detectado = "NINGUNO"
                            for gesto, modelo in modelos.items():
                                prediccion = modelo.predict(input_tensor)[0][0]
                                if prediccion > 0.5:
                                    gesto_detectado = gesto.upper()
                                    break
                        else:
                            gesto_detectado = "NO MANO"

        frame_counter += 1

        cv2.putText(frame, f"Gesto: {gesto_detectado}", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Gesto + Mirada", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detectar_mirada_y_gestos()
