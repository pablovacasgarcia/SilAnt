import mediapipe as mp # importo esta librería aquí porque si no me da error si lo importo el último
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
import cv2
import sys
import subprocess
import threading
import requests
# import mediapipe as mp

# MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Variables globales
gesto_detectado = "NO DETECTADO"
gesto_anterior = None
menu_activo = False
mostrar_menu = False

# Acciones por gesto
import functions  # Asegúrate de tener este módulo implementado
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
        if gesto == "UP":
            functions.abrir_youtube()
            menu_activo = False
        elif gesto == "OPEN":
            functions.abrir_spotify()
            menu_activo = False
        elif gesto == "THREE":
            menu_activo = False
        return

    if gesto != gesto_anterior:
        if gesto == "OPEN":
            menu_activo = True
            mostrar_menu = True
        else:
            accion = acciones_por_gesto.get(gesto)
            if accion:
                accion()
        gesto_anterior = gesto

def enviar_imagen(frame, callback_actualizar_texto):
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
            callback_actualizar_texto(gesto_detectado)
        else:
            print("Error:", response.status_code)
    except Exception as e:
        print("Excepción al enviar imagen:", e)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Control por Gestos")
        self.setGeometry(100, 100, 640, 480)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        self.btn_spotify = QPushButton("Abrir Spotify")
        layout.addWidget(self.btn_spotify)
        self.btn_spotify.clicked.connect(self.abrir_spotify)

        self.lbl_camara = QLabel()
        layout.addWidget(self.lbl_camara)

        self.lbl_gesto = QLabel("Gesto: NO DETECTADO")
        layout.addWidget(self.lbl_gesto)

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: No se pudo acceder a la cámara")
            sys.exit()

        self.timer = QTimer()
        self.timer.timeout.connect(self.actualizar_frame)
        self.timer.start(30)

        self.frame_counter = 0
        self.intervalo_prediccion = 50

    def actualizar_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.lbl_camara.setPixmap(QPixmap.fromImage(q_img))

            # Detección de rostro con MediaPipe
            resultado = face_mesh.process(rgb_frame)
            mirando = False

            if resultado.multi_face_landmarks:
                for landmarks in resultado.multi_face_landmarks:
                    left_eye = landmarks.landmark[159]
                    right_eye = landmarks.landmark[386]
                    nose = landmarks.landmark[1]
                    dx = abs(nose.x - ((left_eye.x + right_eye.x) / 2))
                    if dx < 0.02:
                        mirando = True

            # Si está mirando y toca predecir
            self.frame_counter += 1
            if mirando and self.frame_counter % self.intervalo_prediccion == 0:
                self.frame_counter = 0
                threading.Thread(target=enviar_imagen, args=(frame.copy(), self.actualizar_gesto_texto)).start()

    def actualizar_gesto_texto(self, gesto):
        print("Gesto: " + gesto)
        self.lbl_gesto.setText(f"Gesto: {gesto}")

    def abrir_spotify(self):
        try:
            subprocess.Popen(["spotify"])
        except Exception as e:
            print(f"Error: {e}")

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec())
