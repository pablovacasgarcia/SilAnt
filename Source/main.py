import mediapipe as mp  # Importado al inicio por compatibilidad; evita errores si se importa más tarde
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QSizePolicy
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
import cv2
import sys
import subprocess
import threading
import functions  # Módulo externo con funciones definidas (como cambiar volumen, pausar, etc.)
import requests

# Inicialización del detector de rostro con MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Variables globales para el control de gestos y menú
gesto_detectado = "NO DETECTADO"
gesto_anterior = None
menu_activo = False
mostrar_menu = False
enviando = False 
procesando_prediccion = False



# Diccionario de acciones asociadas a gestos
acciones_por_gesto = {
    "UP": lambda: functions.cambiar_volumen(0.2),
    "DOWN": lambda: functions.cambiar_volumen(-0.2),
    "STOP": functions.pausar_reproduccion,
    "PREV": functions.cancion_anterior,
    "NEXT": functions.cancion_siguiente
}

# Función que ejecuta una acción según el gesto detectado
def ejecutar_accion_por_gesto(gesto):
    global gesto_anterior, menu_activo, mostrar_menu

    if menu_activo:
        # Si el menú está activo, los gestos tienen otras funciones
        if gesto == "UP":
            functions.abrir_youtube()
            menu_activo = False
        elif gesto == "OPEN":
            functions.abrir_spotify()
            menu_activo = False
        elif gesto == "THREE":
            menu_activo = False
        return

    # Evita repetir acciones si el gesto no ha cambiado
    if gesto != gesto_anterior:
        if gesto == "OPEN":
            menu_activo = True
            mostrar_menu = True
        else:
            accion = acciones_por_gesto.get(gesto)
            if accion:
                accion()
        gesto_anterior = gesto

# Envía una imagen capturada al servidor para predecir el gesto
def enviar_imagen(frame, callback_actualizar_texto):
    global gesto_detectado, procesando_prediccion
    _, img_encoded = cv2.imencode('.jpg', frame)
    if procesando_prediccion:
        return
    
    procesando_prediccion = True  # Marca como ocupado
    try:
        response = requests.post(
            'http://localhost:8000/predecir/gesto/',
            files={'img': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')}
        )
        if response.status_code == 200:
            data = response.json()
            gesto_detectado = data.get('gesto', 'NO DETECTADO')
            print("Gesto detectado: ", gesto_detectado)
            ejecutar_accion_por_gesto(gesto_detectado)
            callback_actualizar_texto(gesto_detectado)
        else:
            print("Error:", response.status_code)
    except Exception as e:
        print("Excepción al enviar imagen:", e)
    finally:
        procesando_prediccion = False  # Libera el flag al final


def mostrar_menu_en_frame(frame):
    cv2.putText(frame, "Abrir aplicacion:", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame, "1 -> YouTube", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, "2 -> Spotify", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, "3 -> Cerrar menu", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

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

# Clase principal de la interfaz gráfica
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SilAnt")
        self.setGeometry(100, 100, 640, 480)
        self.setMinimumSize(640, 480)  # Tamaño mínimo de la ventana
        self.setMaximumSize(1920, 1080) # Tamaño máximo de la ventana

        # Configuración del layout principal
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # Etiqueta para mostrar la imagen de la cámara
        self.lbl_camara = QLabel()
        self.lbl_camara.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.lbl_camara.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_camara)

        # Etiqueta para mostrar el gesto detectado
        self.lbl_gesto = QLabel("Gesto: NO DETECTADO")
        layout.addWidget(self.lbl_gesto)

        # Inicialización de la cámara
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: No se pudo acceder a la cámara")
            sys.exit()

        # Temporizador para actualizar los frames de la cámara
        self.timer = QTimer()
        self.timer.timeout.connect(self.actualizar_frame)
        self.timer.start(30)  # Cada 30ms

        self.frame_counter = 0
        self.intervalo_prediccion = 50  # Intervalo de frames para enviar imagen al servidor
    # Método que actualiza los frames de la cámara y realiza predicción si corresponde
    def actualizar_frame(self):
        global mostrar_menu, contador
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)  # Imagen espejo
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detección facial con MediaPipe
            resultado = face_mesh.process(rgb_frame)
            mirando = False

            if resultado.multi_face_landmarks:
                for landmarks in resultado.multi_face_landmarks:
                    left_eye = landmarks.landmark[159]
                    right_eye = landmarks.landmark[386]
                    nose = landmarks.landmark[1]
                    dx = abs(nose.x - ((left_eye.x + right_eye.x) / 2))
                    if dx < 0.02:  # Umbral para determinar si está mirando de frente
                        mirando = True

            # Si está mirando de frente y es momento de predecir
            self.frame_counter += 1
            
            if mirando and self.frame_counter % self.intervalo_prediccion == 0:
                self.frame_counter = 0
                mano = recortar_mano(rgb_frame)
                if mano is not None: 
                    # Ejecutar predicción en hilo aparte
                    threading.Thread(target=enviar_imagen, args=(mano.copy(), self.actualizar_gesto_texto)).start()


            if mostrar_menu:
                mostrar_menu_en_frame(rgb_frame)
                if not menu_activo:
                    mostrar_menu = False  # Oculta menú si ya no está activo

            # Dibuja un punto si se detecta que está mirando de frente
            if mirando:
                alto, ancho, _ = rgb_frame.shape
                centro_x = ancho // 2
                centro_y = alto - 50
                cv2.circle(rgb_frame, (centro_x, centro_y), 10, (0, 255, 0), -1)
                
            # Mostrar imagen en interfaz
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w

            q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            scaled_pixmap = pixmap.scaled(self.lbl_camara.size(), Qt.AspectRatioMode.KeepAspectRatio)
            self.lbl_camara.setPixmap(scaled_pixmap)
            # self.lbl_camara.setPixmap(QPixmap.fromImage(q_img))



            

    # Actualiza la etiqueta de texto con el gesto detectado
    def actualizar_gesto_texto(self, gesto):
        print("Gesto: " + gesto)
        self.lbl_gesto.setText(f"Gesto: {gesto}")

    # Libera la cámara al cerrar la ventana
    def closeEvent(self, event):
        self.cap.release()
        event.accept()

# Inicialización de la aplicación
app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec())
