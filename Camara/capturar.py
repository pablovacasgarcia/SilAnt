import cv2
import mediapipe as mp

"""
    Este script detecta si el usuario está mirando a la cámara y si tiene la mano en frente de la cámara.
    Para detectar la mirada, se utilizan los puntos de los ojos y la nariz.
    Para detectar la mano, se utilizan los puntos de la mano.
"""
def detectar_mirada_y_mano():
    # Inicializar los modelos de detección de rostro
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Inicializar los modelos de detección de manos
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Inicializar la cámara
    cap = cv2.VideoCapture(0)
    
    # Leer la cámara
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Voltear la imagen horizontalmente
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Procesar rostro
        face_result = face_mesh.process(rgb_frame)
        # Procesar manos
        hand_result = hands.process(rgb_frame)

        mirando = False  # Bandera para saber si está mirando a la cámara

        if face_result.multi_face_landmarks:
            for face_landmarks in face_result.multi_face_landmarks:
                left_eye = face_landmarks.landmark[159]
                right_eye = face_landmarks.landmark[386]
                nose = face_landmarks.landmark[1]
                
                dx = abs(nose.x - ((left_eye.x + right_eye.x) / 2))
                
                if dx < 0.02:  # Está mirando a la cámara
                    mirando = True
                    cv2.putText(frame, "Mirando a la camara", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Solo detectar la mano si está mirando a la cámara
        if mirando and hand_result.multi_hand_landmarks:
            for hand_landmarks in hand_result.multi_hand_landmarks:
                h, w, _ = frame.shape
                x_coords = []
                y_coords = []

                # Guardar todas las coordenadas de los puntos de la mano
                for landmark in hand_landmarks.landmark:
                    x_coords.append(int(landmark.x * w))
                    y_coords.append(int(landmark.y * h))

                # Obtener las coordenadas extremas para cubrir toda la mano
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                # AÑADIR UN MARGEN ADICIONAL DEL 20%
                margen_x = int((x_max - x_min) * 0.2)
                margen_y = int((y_max - y_min) * 0.2)

                x_min = max(0, x_min - margen_x)
                x_max = min(w, x_max + margen_x)
                y_min = max(0, y_min - margen_y)
                y_max = min(h, y_max + margen_y)

                # Dibujar un rectángulo alrededor de la mano
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

        cv2.imshow('Detección de mirada y mano', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detectar_mirada_y_mano()
