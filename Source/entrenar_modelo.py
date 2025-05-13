from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os
import numpy as np
import cv2
import random
import shutil
import matplotlib.pyplot as plt   

class CrearModelo:
    modelo = None
    history = None
    def __init__(self, shape=(48, 48, 1)):
        print("Estoy entrenando un nuevo modelo")
        self.modelo = Sequential()
        self.modelo.add(Input(shape=shape))
        self.modelo.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        self.modelo.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        self.modelo.add(MaxPooling2D(pool_size=(2, 2)))
        self.modelo.add(Dropout(0.25))

        self.modelo.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.modelo.add(MaxPooling2D(pool_size=(2, 2)))
        self.modelo.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.modelo.add(MaxPooling2D(pool_size=(2, 2)))
        self.modelo.add(Dropout(0.25))

        self.modelo.add(Flatten())
        
        self.modelo.add(Dense(1024, activation='relu'))
        self.modelo.add(Dropout(0.5))
        self.modelo.add(Dense(1, activation='sigmoid'))

        self.modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def entrenar_modelo(self, nombre_modelo, target_size, gesto=None, batch_size=32, epochs=10):
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.05,
            zoom_range=0.1,
            horizontal_flip=False,
            fill_mode='nearest'
        )


        val_datagen = ImageDataGenerator(
            rescale=1./255
        )

        train_generator = train_datagen.flow_from_directory(
            os.path.join("Dataset", "train"),
            target_size=target_size,
            batch_size=batch_size,
            color_mode='grayscale',
            class_mode='binary'
        )

        val_generator = val_datagen.flow_from_directory(
            os.path.join("Dataset", 'validate'),
            target_size=target_size,
            batch_size=batch_size,
            color_mode='grayscale',
            class_mode='binary'
        )

        print(self.modelo.summary())

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            f"./Modelos/{nombre_modelo}/modelo_{gesto.lower()}.h5",
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        reduccion_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=5, # modificar según las épocas
            min_lr=0.00001
        )
        
        parada_temprana = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5, # modificar según las épocas
            restore_best_weights=True
        )

        self.history = self.modelo.fit(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=len(val_generator),
            callbacks=[checkpoint, reduccion_lr, parada_temprana]
        )

        if gesto == None:
            print("⚠️ No se ha especificado un gesto para guardar el modelo.")
            return

        self.modelo.save(f"./Modelos/{nombre_modelo}/modelo_{gesto.lower()}_final.h5")
    


    
    def estructura_directorios(self, gesto=None):
        """
            A partir de un gesto, prepara una estructura de directorios diviendo las imágenes en entrenamiento y validación.
        """
        categorias = os.listdir("./Images/train")
        random.seed(42)  # Para reproducibilidad

        # Crear directorios de entrenamiento y validación
        os.makedirs(os.path.join("./Dataset", "train", "1"), exist_ok=True)
        os.makedirs(os.path.join("./Dataset", "train", "0"), exist_ok=True)
        os.makedirs(os.path.join("./Dataset", "validate", "1"), exist_ok=True)
        os.makedirs(os.path.join("./Dataset", "validate", "0"), exist_ok=True)

        contador_train_0 = 0
        contador_train_1 = 0
        contador_val_0 = 0
        contador_val_1 = 0
        for categoria in categorias:
            carpeta_original = os.path.join("Images", "train", categoria)
            imagenes = [f for f in os.listdir(carpeta_original) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
            if not imagenes:
                print(f"⚠️ No hay imágenes en {carpeta_original}")
                continue

            # Mezclar y dividir datos
            random.shuffle(imagenes)
            split_idx = int(len(imagenes) * 0.8)
            train_files = imagenes[:split_idx]
            validate_files = imagenes[split_idx:]

            # Mover archivos a sus respectivas carpetas
            for file in train_files:
                extension = file.split(".")[1]
                if categoria.lower() == gesto.lower():
                    shutil.copy(os.path.join(carpeta_original, file), os.path.join("Dataset", "train", "1", f"{contador_train_1}.{extension}"))
                    contador_train_1 += 1
                else:
                    shutil.copy(os.path.join(carpeta_original, file), os.path.join("Dataset", "train", "0", f"{contador_train_0}.{extension}"))
                    contador_train_0 += 1          

            for file in validate_files:
                if categoria.lower() == gesto.lower():
                    shutil.copy(os.path.join(carpeta_original, file), os.path.join("Dataset", "validate", "1", f"{contador_val_1}.{extension}"))
                    contador_val_1 += 1
                else:
                    shutil.copy(os.path.join(carpeta_original, file), os.path.join("Dataset", "validate", "0", f"{contador_val_0}.{extension}"))
                    contador_val_0 += 1

            print(f"✅ {categoria}: {len(train_files)} train, {len(validate_files)} validate")

    def guardar_grafico_loss_val_loss(self, nombre_modelo, gesto):
        ruta = os.path.join("Images", "modelos", nombre_modelo)
        os.makedirs(ruta, exist_ok=True)

        loss = self.history.history["loss"]
        val_loss = self.history.history["val_loss"]
        accuracy = self.history.history["accuracy"]
        val_accuracy = self.history.history["val_accuracy"]
        epocas = range(1, len(loss) + 1)

        # Gráfico de pérdida
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(epocas, loss, 'b-', label='Training Loss')
        ax1.plot(epocas, val_loss, 'r-', label='Validation Loss')
        ax1.set_title('Pérdida (Loss)')
        ax1.set_xlabel('Épocas')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        fig1.savefig(f"{ruta}/loss_modelo_{gesto.lower()}.png", dpi=300, bbox_inches='tight')

        # Gráfico de precisión
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(epocas, accuracy, 'g-', label='Training Accuracy')
        ax2.plot(epocas, val_accuracy, 'orange', label='Validation Accuracy')
        ax2.set_title('Precisión (Accuracy)')
        ax2.set_xlabel('Épocas')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        fig2.savefig(f"{ruta}/accuracy_modelo_{gesto.lower()}.png", dpi=300, bbox_inches='tight')

        print(f"✅ Imágenes guardadas: loss y accuracy en {ruta}")

def evaluar_modelo(nombre_modelo, gesto, target_size, reshape):
    """
    Evalúa el modelo entrenado con un conjunto de datos de prueba.
    """
    X = []         # imágenes procesadas
    y = []         # 1 si la imagen es 'up', 0 en caso contrario

    for archivo in os.listdir('./Images/Pruebas'):
        if archivo.endswith('.jpg'):
            ruta = os.path.join('Images', 'Pruebas', archivo)
            img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, target_size) / 255.0
            img = img.reshape(reshape)
            X.append(img)

            # Etiqueta binaria
            etiqueta = 1 if archivo.split('-')[0] == gesto else 0
            y.append(etiqueta)

    X = np.array(X)
    y = np.array(y)
    modelo_up = load_model(f"./Modelos/{nombre_modelo}/modelo_{gesto.lower()}_final.h5")
    loss, accuracy = modelo_up.evaluate(X, y)
    print(f'Evaluando modelo {gesto.upper()}: Pérdida: {loss:.4f}, Precisión: {accuracy:.4f}')
  
        
if __name__ == "__main__":
    # Nombre que se le va a dar a los directorios. 
    nombre_modelo = "109"

    gestos = os.listdir("./Images/train")
    ruta = os.path.join("./Modelos", nombre_modelo)
    if os.path.exists(ruta):
        raise Exception("Ese directorio ya existe. Elige otro nombre")
    os.makedirs(os.path.join("./Modelos", nombre_modelo))

    epocas = 15
    target_size = (48, 48)
    shape=(48, 48, 1)
    for gesto in gestos:
        print("------------------------------", end="\n\n")
        print(f"Entrenando el modelo: {gesto.upper()}", end="\n\n")
        print("------------------------------")

        modelo = CrearModelo(shape=shape)
        modelo.estructura_directorios(gesto)
        modelo.entrenar_modelo(nombre_modelo=nombre_modelo, target_size=target_size, gesto=gesto, batch_size=32, epochs=epocas)
        modelo.guardar_grafico_loss_val_loss(nombre_modelo, gesto)
        evaluar_modelo(nombre_modelo=nombre_modelo, gesto=gesto, target_size=target_size, reshape=shape)


    
    # Evaluar un modelo sin realizar los pasos de entrenamiento
    """
    nombre_modelo = "3"
    gestos = os.listdir("./Images/train")
    for gesto in gestos:
        evaluar_modelo(nombre_modelo=nombre_modelo, gesto=gesto)
    """
