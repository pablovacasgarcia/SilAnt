from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import pyautogui
import os
import sys
import subprocess
import webbrowser


def cambiar_volumen(ajuste):
    """
        Función para cambiar el volumen del sistema
        :param ajuste: Ajuste de volumen, rango de -1.0 a 1.0
    """

    # Obtener interfaz de volumen
    dispositivos = AudioUtilities.GetSpeakers()
    interfaz = dispositivos.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volumen = cast(interfaz, POINTER(IAudioEndpointVolume))

    # Obtener volumen actual (rango 0.0 - 1.0)
    volumen_actual = volumen.GetMasterVolumeLevelScalar()
    nuevo_volumen = max(0.0, min(1.0, volumen_actual + ajuste))

    # Cambiar volumen
    volumen.SetMasterVolumeLevelScalar(nuevo_volumen, None)
    print(f"Nuevo volumen: {nuevo_volumen * 100:.2f}%")

# Subir volumen en un 5%                
# cambiar_volumen(0.05)

# Para bajar volumen en un 5%, usa:     
# cambiar_volumen(-0.05)

# Para silenciar,                       
# cambiar_volumen(-1.0)


def pausar_reproduccion():
    """"
        Función para pausar la reproducción de música o video
    """

    pyautogui.press("playpause")  # Envía la tecla multimedia "Play/Pause"
    print("Reproducción pausada.")

# Ejecutar la función,                  
# pausar_reproduccion()


def cancion_anterior():
    """
        Función para reproducir la canción o video anterior
    """

    pyautogui.press("prevtrack")  # Simula la tecla multimedia "anterior"
    print("Canción/video anterior.")

# Ejecutar función
# cancion_anterior()


def cancion_siguiente():
    """
        Función para reproducir la siguiente canción o video
    """

    pyautogui.press("nexttrack")  # Simula la tecla multimedia "siguiente"
    print("Canción/video siguiente.")

# Ejecutar función
# cancion_siguiente()


def abrir_spotify():
    """
        Función para abrir la aplicación de Spotify"
    """

    try:
        if sys.platform.startswith("win"):
            # Windows: Ejecutar Spotify desde la ruta estándar
            subprocess.run(["start", "spotify"], shell=True)
        elif sys.platform.startswith("darwin"):
            # macOS: Abrir la aplicación Spotify
            subprocess.run(["open", "-a", "Spotify"])
        elif sys.platform.startswith("linux"):
            # Linux: Ejecutar Spotify desde el terminal
            subprocess.run(["spotify"])
        print("Spotify abierto correctamente.")
    except Exception as e:
        print(f"Error al abrir Spotify: {e}")

# Ejecutar la función
# abrir_spotify()


def abrir_youtube():
    """
        Función para abrir YouTube en el navegador
    """
    
    url = "https://www.youtube.com"
    webbrowser.open(url)
    print("YouTube abierto en el navegador.")

# Ejecutar la función
# abrir_youtube()
