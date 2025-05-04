# Proyecto Final: Detección de Gestos y Mirada

Este proyecto implementa un sistema de detección de gestos y mirada utilizando modelos de aprendizaje profundo y herramientas de visión por computadora. Incluye una API basada en Django para predicciones de gestos y una aplicación para la detección en tiempo real.


## **Estructura del Proyecto**

```
IABDProyectoFinal/
├── api_gestos/          # API en Django para predicción de gestos
├── Images/              # Imágenes de entrenamiento y modelos
├── Modelos/             # Modelos entrenados en formato .h5
├── Source/              # Scripts para procesamiento, entrenamiento y la aplicación.
└── README.md            # Documentación del proyecto
```

---

## **Requisitos**

### **Dependencias**
- Python 3.12.9 
- Librerías de Python:
  - Django
  - Flask
  - TensorFlow
  - OpenCV
  - Mediapipe
  - Matplotlib
  - NumPy
  - Pillow
  - scikit-learn
- Otros:
  - Navegador web para la interfaz de usuario.

### **Instalación**
1. Clona este repositorio:
```bash
git clone https://github.com/pablovacasgarcia/SilAnt.git
cd SilAnt
```

2. Crea un entorno virtual
```bash 
python -m venv venv' 
```

3. Activa el entorno virtual.
- Windows.
```bash 
.\venv\Scripts\activate 
```

- Linux/Mac
```bash 
source ./venv/bin/activate 
```

4. Instala las dependencias:
```bash
 pip install -r requirements.txt
```

---

## **Uso**

### **API de Predicción de Gestos**
1. Primero iniciamos la API, que recogerá los frames y devolverá una respuesta con el gesto en JSON:
```bash
cd api_gestos
python manage.py runserver
```

2. Abrimos una nueva terminal e iniciamos el script para probar la API:
```bash
python ./Source/app.py
```

3. Se abrirá el programa, que detecta cuando miramos a la cámara, mostrando un círculo verde, para analizar los gestos, recogerá los frames y enviará la petición a la API que devolverá el siguiente resultado, cada gesto activará una función determinada:
```json
{
      "gesto": "Gesto detectado"
}
```

---

## **Consideraciones**
- Los modelos entrenados están en la carpeta Modelos y configurados correctamente en los scripts.
- Asegúrate de que las imágenes de entrenamiento estén organizadas en `/Image/train`.
- Configura las rutas de los modelos en los scripts de predicción en `/Modelos/XX`.

---

## **Credenciales**
No se requieren credenciales específicas para este proyecto. Sin embargo, asegúrate de configurar correctamente las rutas y permisos de los archivos.

---

## **Autores**
- Pablo Vacas García.
- Juan Manuel García Moyano.
- Ángel Talavera Garrido.
