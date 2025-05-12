from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import pyautogui
import gc
import sys
import subprocess
import webbrowser

import comtypes 

def cambiar_volumen(ajuste):
    """
    Función para cambiar el volumen del sistema. Maneja correctamente hilos con COM.
    """
    comtypes.CoInitialize()  # Inicializa COM para este hilo
    volumen = None
    try:
        dispositivos = AudioUtilities.GetSpeakers()
        interfaz = dispositivos.Activate(
            IAudioEndpointVolume._iid_, CLSCTX_ALL, None
        )
        volumen = cast(interfaz, POINTER(IAudioEndpointVolume))

        volumen_actual = volumen.GetMasterVolumeLevelScalar()
        if volumen_actual + ajuste >= 1.0:
            nuevo_volumen = 1.0

        elif volumen_actual + ajuste <= 0.0:
            nuevo_volumen = 0.0
        else:  
            nuevo_volumen = max(0.0, min(1.0, volumen_actual + ajuste))

        volumen.SetMasterVolumeLevelScalar(nuevo_volumen, None)
        print(f"Nuevo volumen: {nuevo_volumen * 100:.2f}%")
    except Exception as e:
        print("Error al cambiar volumen:", e)
    finally:
        volumen = None  # Elimina referencias
        gc.collect()    # Fuerza la liberación
        comtypes.CoUninitialize()  # Libera COM



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
