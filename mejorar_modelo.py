import os, random
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

DATASET_LISTO = "dataset_listo"
CATS = ["Carton", "Metal", "Papel", "Plastico", "Vidrio"]
IMG_SIZE = (224, 224)
BATCH = 32
MODELO_PATH = "modelo_residuos.h5"
random.seed(42); np.random.seed(42); tf.random.set_seed(42)

def construir_modelo():
    base = MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet")
    base.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(512, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    out = Dense(len(CATS), activation="softmax")(x)
    m = Model(inputs=base.input, outputs=out)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss="categorical_crossentropy", metrics=["accuracy"])
    return m

aug = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.25,
    height_shift_range=0.25,
    shear_range=0.2,
    zoom_range=0.3,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode="nearest"
)
vgen = ImageDataGenerator(preprocessing_function=preprocess_input)

tr = aug.flow_from_directory(os.path.join(DATASET_LISTO,"train"),
    target_size=IMG_SIZE, batch_size=BATCH, class_mode="categorical")
vl = vgen.flow_from_directory(os.path.join(DATASET_LISTO,"val"),
    target_size=IMG_SIZE, batch_size=BATCH, class_mode="categorical")

print("Orden:", tr.class_indices)

modelo = construir_modelo()
cb = [
    ModelCheckpoint(MODELO_PATH, save_best_only=True, monitor="val_accuracy", verbose=1),
    EarlyStopping(patience=7, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.3, patience=3, min_lr=1e-8, verbose=1)
]

print("\n== FASE 1: Cabeza mejorada ==")
modelo.fit(tr, validation_data=vl, epochs=20, callbacks=cb)

print("\n== FASE 2: Fine-tuning profundo ==")
for layer in modelo.layers:
    layer.trainable = True
for layer in modelo.layers[:-50]:
    layer.trainable = False
modelo.compile(optimizer=tf.keras.optimizers.Adam(5e-6),
               loss="categorical_crossentropy", metrics=["accuracy"])
modelo.fit(tr, validation_data=vl, epochs=15, callbacks=cb)
modelo.save(MODELO_PATH)
print(f"\nModelo mejorado guardado: {MODELO_PATH}")
