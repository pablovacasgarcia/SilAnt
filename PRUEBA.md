# Proyecto Final: Detección de Gestos y Mirada

Este proyecto implementa un sistema de detección de gestos y mirada utilizando modelos de aprendizaje profundo y herramientas de visión por computadora. Incluye una API basada en Django para predicciones de gestos y una aplicación web con Flask para la detección en tiempo real.


## **Estructura del Proyecto**

```
IABDProyectoFinal/
├── api_gestos/          # API en Django para predicción de gestos
├── Images/              # Imágenes de entrenamiento y modelos
├── Modelos/             # Modelos entrenados en formato .h5
├── Source/              # Scripts para entrenamiento y procesamiento
├── Web/                 # Aplicación web con Flask
└── README.md            # Documentación del proyecto
```

---

## **Requisitos**

### **Dependencias**
- Python 3.8+
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
  - Flask-Cors
- Otros:
  - Navegador web para la interfaz de usuario.

### **Instalación**
1. Clona este repositorio:
   ```bash
   git clone https://github.com/tu_usuario/IABDProyectoFinal.git
   cd IABDProyectoFinal
   ```

2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

3. Configura las credenciales y rutas en los archivos `settings.py` (para Django) y `app.py` (para Flask).

4. Ejecuta las migraciones de Django:
   ```bash
   python manage.py migrate
   ```

5. Inicia los servidores:
   - **API de gestos (Django):**
     ```bash
     python manage.py runserver
     ```
   - **Aplicación web (Flask):**
     ```bash
     python Web/app.py
     ```

---

## **Uso**

### **API de Predicción de Gestos**
1. Envía una solicitud POST a `/api/predict/` con una imagen en el campo `image`.
   ```bash
   curl -X POST -F "image=@path_to_image.jpg" http://127.0.0.1:8000/api/predict/
   ```

2. Recibirás una respuesta JSON con el resultado de la predicción:
   ```json
   {
       "prediction": "Gesto detectado"
   }
   ```

### **Aplicación Web**
1. Accede a la interfaz web en `http://127.0.0.1:5000/`.
2. Visualiza el video en tiempo real con detección de mirada y gestos.

---

## **Consideraciones**
- Los modelos entrenados deben estar en la carpeta Modelos y configurados correctamente en los scripts.
- Asegúrate de que las imágenes de entrenamiento y validación estén organizadas en train y `Images/validate`.
- Configura las rutas de los modelos en los scripts de predicción (`api_gestos/views.py` y app.py).

---

## **Credenciales**
No se requieren credenciales específicas para este proyecto. Sin embargo, asegúrate de configurar correctamente las rutas y permisos de los archivos.

---

## **Autores**
- Nombre del equipo o autor principal.
- Contacto: [correo@example.com](mailto:correo@example.com)
```

---

### **Informe Técnico**

#### **Introducción**
El objetivo del proyecto es desarrollar un sistema que permita detectar gestos y mirada en tiempo real utilizando modelos de aprendizaje profundo y herramientas de visión por computadora. Este sistema tiene aplicaciones en interfaces hombre-máquina, accesibilidad y control por gestos.

---

#### **Metodología**
1. **Preparación de Datos**:
   - Las imágenes se organizaron en carpetas para entrenamiento y validación (`Images/train` y `Images/validate`).
   - Se implementaron scripts para dividir los datos y preprocesarlos (`entrenar_modelo.py`).

2. **Entrenamiento de Modelos**:
   - Se utilizaron redes neuronales convolucionales (CNN) implementadas con TensorFlow/Keras.
   - Los modelos se entrenaron para detectar gestos específicos y se guardaron en la carpeta `Modelos/`.

3. **Desarrollo de la API**:
   - Se utilizó Django REST Framework para crear una API que recibe imágenes y devuelve predicciones.
   - La API carga los modelos entrenados y realiza predicciones en tiempo real.

4. **Aplicación Web**:
   - Flask se utilizó para desarrollar una interfaz web que muestra la detección de mirada y gestos en tiempo real.
   - Mediapipe y OpenCV se emplearon para el procesamiento de video.

---

#### **Decisiones Tomadas**
- **Frameworks**:
  - Django se eligió para la API debido a su robustez y facilidad para manejar solicitudes HTTP.
  - Flask se utilizó para la aplicación web por su simplicidad y flexibilidad.
- **Modelos**:
  - Se optó por TensorFlow/Keras debido a su amplio soporte y facilidad de uso.
- **Procesamiento de Video**:
  - Mediapipe se utilizó para la detección de características faciales debido a su precisión y rendimiento.

---

#### **Resultados**
- **API**:
  - La API puede procesar imágenes y devolver predicciones con una precisión aceptable.
- **Aplicación Web**:
  - La interfaz web permite visualizar la detección de mirada y gestos en tiempo real.
- **Modelos**:
  - Los modelos entrenados alcanzaron una precisión promedio del 85% en el conjunto de validación.

---

#### **Conclusiones**
- El sistema desarrollado cumple con los objetivos planteados, proporcionando una solución funcional para la detección de gestos y mirada.
- **Ventajas**:
  - Modularidad: La API y la aplicación web pueden funcionar de manera independiente.
  - Escalabilidad: Se pueden agregar nuevos gestos entrenando modelos adicionales.
- **Limitaciones**:
  - La precisión depende de la calidad de los datos de entrenamiento.
  - El rendimiento en tiempo real puede verse afectado en hardware de baja potencia.
- **Futuras Mejoras**:
  - Optimización de los modelos para dispositivos móviles.
  - Ampliación del conjunto de datos para incluir más gestos y condiciones de iluminación.

---

#### **Anexos**
- **Código Fuente**:
  - Disponible en el repositorio de GitHub: [https://github.com/tu_usuario/IABDProyectoFinal](https://github.com/tu_usuario/IABDProyectoFinal)
- **Instrucciones de Despliegue**:
  - Incluidas en el archivo `README.md`.

---

Con este informe y el archivo `README.md`, tienes toda la documentación necesaria para publicar y poner en funcionamiento el proyecto. Si necesitas ayuda adicional, no dudes en pedírmelo.---

### **Informe Técnico**

#### **Introducción**
El objetivo del proyecto es desarrollar un sistema que permita detectar gestos y mirada en tiempo real utilizando modelos de aprendizaje profundo y herramientas de visión por computadora. Este sistema tiene aplicaciones en interfaces hombre-máquina, accesibilidad y control por gestos.

---

#### **Metodología**
1. **Preparación de Datos**:
   - Las imágenes se organizaron en carpetas para entrenamiento y validación (`Images/train` y `Images/validate`).
   - Se implementaron scripts para dividir los datos y preprocesarlos (`entrenar_modelo.py`).

2. **Entrenamiento de Modelos**:
   - Se utilizaron redes neuronales convolucionales (CNN) implementadas con TensorFlow/Keras.
   - Los modelos se entrenaron para detectar gestos específicos y se guardaron en la carpeta `Modelos/`.

3. **Desarrollo de la API**:
   - Se utilizó Django REST Framework para crear una API que recibe imágenes y devuelve predicciones.
   - La API carga los modelos entrenados y realiza predicciones en tiempo real.

4. **Aplicación Web**:
   - Flask se utilizó para desarrollar una interfaz web que muestra la detección de mirada y gestos en tiempo real.
   - Mediapipe y OpenCV se emplearon para el procesamiento de video.

---

#### **Decisiones Tomadas**
- **Frameworks**:
  - Django se eligió para la API debido a su robustez y facilidad para manejar solicitudes HTTP.
  - Flask se utilizó para la aplicación web por su simplicidad y flexibilidad.
- **Modelos**:
  - Se optó por TensorFlow/Keras debido a su amplio soporte y facilidad de uso.
- **Procesamiento de Video**:
  - Mediapipe se utilizó para la detección de características faciales debido a su precisión y rendimiento.

---

#### **Resultados**
- **API**:
  - La API puede procesar imágenes y devolver predicciones con una precisión aceptable.
- **Aplicación Web**:
  - La interfaz web permite visualizar la detección de mirada y gestos en tiempo real.
- **Modelos**:
  - Los modelos entrenados alcanzaron una precisión promedio del 85% en el conjunto de validación.

---

#### **Conclusiones**
- El sistema desarrollado cumple con los objetivos planteados, proporcionando una solución funcional para la detección de gestos y mirada.
- **Ventajas**:
  - Modularidad: La API y la aplicación web pueden funcionar de manera independiente.
  - Escalabilidad: Se pueden agregar nuevos gestos entrenando modelos adicionales.
- **Limitaciones**:
  - La precisión depende de la calidad de los datos de entrenamiento.
  - El rendimiento en tiempo real puede verse afectado en hardware de baja potencia.
- **Futuras Mejoras**:
  - Optimización de los modelos para dispositivos móviles.
  - Ampliación del conjunto de datos para incluir más gestos y condiciones de iluminación.

---

#### **Anexos**
- **Código Fuente**:
  - Disponible en el repositorio de GitHub: [https://github.com/tu_usuario/IABDProyectoFinal](https://github.com/tu_usuario/IABDProyectoFinal)
- **Instrucciones de Despliegue**:
  - Incluidas en el archivo `README.md`.

---

Con este informe y el archivo `README.md`, tienes toda la documentación necesaria para publicar y poner en funcionamiento el proyecto. Si necesitas ayuda adicional, no dudes en pedírmelo.