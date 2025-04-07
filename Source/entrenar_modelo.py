from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os
import pandas as pd
import numpy as np
from PIL import Image
import random
import shutil

class CrearModelo:
    modelo = None
    def __init__(self):
        self.modelo = Sequential()

        self.modelo.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
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

    def entrenar_modelo(self, gesto=None, batch_size=32, epochs=10):
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        val_datagen = ImageDataGenerator(
            rescale=1./255
        )

        train_generator = train_datagen.flow_from_directory(
            os.path.join("Dataset", "train"),
            target_size=(48, 48),
            batch_size=batch_size,
            color_mode='grayscale',
            class_mode='binary'
        )

        val_generator = val_datagen.flow_from_directory(
            os.path.join("Dataset", 'validate'),
            target_size=(48, 48),
            batch_size=batch_size,
            color_mode='grayscale',
            class_mode='binary'
        )

        print(self.modelo.summary())

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            f"modelo_{gesto.lower()}.h5",
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        reduccion_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=10,
            min_lr=0.00001
        )
        
        parada_temprana = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )

        self.modelo.fit(
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

        self.modelo.save(f"modelo_{gesto.lower()}_final.h5")
    


    
    def estructura_directorios(self, gesto=None):
        categorias = os.listdir("./Images")
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
            carpeta_original = os.path.join("Images", categoria)
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
        
        
if __name__ == "__main__":
    gesto = "Down"
    modelo = CrearModelo()
    modelo.estructura_directorios(gesto)
    modelo.entrenar_modelo(gesto=gesto, batch_size=32, epochs=10)
    