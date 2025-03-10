import cv2
import mediapipe as mp

"""
    Este script detecta si el usuario está mirando a la cámara para enviar el frame a la api.
    Para detectar la mirada, se utilizan los puntos de los ojos y la nariz.
"""
def detectar_mirada():
    # Inicializar los modelos de detección de rostro
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

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

        cv2.imshow('Detección de mirada', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detectar_mirada()
