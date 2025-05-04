import cv2
import mediapipe as mp
import requests
import threading

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

gesto_detectado = "NO DETECTADO"

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
            gesto_detectado = data.get('gesto', 'nh')
        else:
            print("Error:", response.status_code)
    except Exception as e:
        print("Excepción al enviar imagen:", e)

def detectar_mirada_y_gestos():
    cap = cv2.VideoCapture(0)
    frame_counter = 0
    prediccion_intervalo = 50  

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
                        threading.Thread(target=enviar_imagen, args=(frame.copy(),)).start()

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
