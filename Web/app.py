from flask import Flask, render_template, Response
import cv2
import mediapipe as mp

app = Flask(__name__)

# Inicializar modelos de Mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def detectar_mirada():
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Procesar rostro
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
                    cv2.putText(frame, "Mirando a la camara", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Codificar el frame en formato JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(detectar_mirada(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)