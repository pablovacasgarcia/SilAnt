# Importación de librerías necesarias
import cv2  # Para manejo y procesamiento de imágenes
import mediapipe as mp  # Para detección de manos mediante landmarks
import numpy as np
import os  # Para interactuar con el sistema de archivos

# Inicialización del detector de manos de MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

def recortar_mano(image, margen=0.2):
    """
    Detecta la mano más alta (la más cercana al borde superior de la imagen) y devuelve un recorte cuadrado de esa zona.
    Se aplica un margen para ampliar el área del recorte.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        return None  # No se detectó ninguna mano

    min_y = float('inf')  # Inicializamos con un valor muy alto
    best_hand_bbox = None
    image_h, image_w, _ = image.shape

    # Buscar la mano más "alta" en la imagen
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

    # Calcular tamaño del recorte cuadrado
    side_length = max(x_max - x_min, y_max - y_min)
    extra_space = int(side_length * margen)
    side_length += extra_space * 2

    # Calcular centro del recorte
    center_x, center_y = (x_min + x_max) // 2, (y_min + y_max) // 2

    # Calcular límites del recorte incluyendo margen, sin salirse de la imagen
    x_min_sq = max(center_x - side_length // 2, 0)
    y_min_sq = max(center_y - side_length // 2, 0)
    x_max_sq = min(center_x + side_length // 2, image_w)
    y_max_sq = min(center_y + side_length // 2, image_h)

    return image[y_min_sq:y_max_sq, x_min_sq:x_max_sq]


def procesar_imagenes(folder_path="Images/recortar", save_path="Images/train", margen=0.2):
    """
    Recorre todas las imágenes en 'folder_path', recorta la mano más alta de cada una,
    y guarda el resultado en la ruta correspondiente dentro de 'save_path', manteniendo la estructura de carpetas.
    """

    formatos_validos = ('.jpg', '.jpeg', '.png')  # Formatos soportados
    contador = -1  # Contador de imágenes dentro de cada subcarpeta
    last_dir = ""  # Última carpeta procesada

    # Recorremos todas las subcarpetas e imágenes
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith(formatos_validos):
                image_path = os.path.join(root, filename)
                image = cv2.imread(image_path)

                if image is None:
                    print(f"Error al cargar: {filename}")
                    continue

                mano_recortada = recortar_mano(image, margen)

                # Obtener la ruta relativa para replicar estructura de carpetas
                relative_path = os.path.relpath(root, folder_path)

                if last_dir != relative_path:
                    contador = -1  # Reiniciar contador al cambiar de carpeta

                if contador == -1 and last_dir != relative_path:
                    last_dir = relative_path  
                    ruta_prueba = os.path.join(*save_path.split("/"), relative_path)
                    os.makedirs(ruta_prueba, exist_ok=True)

                    # Buscar el último número guardado en esa carpeta para continuar
                    prueba = [int(f.split(".")[0]) for f in os.listdir(ruta_prueba) if f.split(".")[0].isdigit()]
                    prueba.sort()
                    contador = sorted(prueba)[-1] + 1 if prueba else 0

                # Crear carpeta de guardado si no existe
                save_dir = os.path.join(save_path, relative_path)
                os.makedirs(save_dir, exist_ok=True)
                extension = filename.split('.')[-1]
                save_image_path = os.path.join(save_dir, f"{contador}.{extension}")

                if mano_recortada is not None:
                    cv2.imwrite(save_image_path, mano_recortada)
                    print(f"✅ Procesado y guardado: {save_image_path}")
                    contador += 1
                else:
                    print(f"❌ No se detectó mano en: {image_path}, no se guarda.")

                os.remove(image_path)  # Borrar la imagen original en cualquier caso


# Ejecución directa del procesamiento si se ejecuta el script
procesar_imagenes()
