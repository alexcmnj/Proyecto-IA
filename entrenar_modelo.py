"""
=============================================================
  ENTRENAMIENTO CNN — CLASIFICADOR DE RESIDUOS
  MobileNetV2 + Transfer Learning + TensorFlow
  Dataset: Garbage Classification (Kaggle)
=============================================================
  Pasos para usar:
    1. Descarga el dataset desde:
       https://www.kaggle.com/datasets/mostafaabla/garbage-classification
    2. Descomprime y deja la carpeta así:
         dataset/
           Garbage classification/
             Garbage classification/
               garbage_classification/
                 cardboard/  glass/  metal/  paper/  plastic/ ...
    3. Ejecuta: python entrenar_modelo.py
    4. Al terminar ejecuta: python proyectoIA.py
=============================================================
"""

import os
import json
import shutil
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


# ═══════════════════════════════════════════════════════════
#  CONFIGURACIÓN — ajusta solo estas variables si es necesario
# ═══════════════════════════════════════════════════════════

# Ruta donde está el dataset descomprimido
DATASET_RAW = "dataset/Garbage classification/Garbage classification/garbage_classification"

# Carpeta que se creará con la estructura train/val
DATASET_LISTO = "dataset_listo"

# Mapeo carpetas del dataset → categorías en español
# El vidrio viene en 3 colores en este dataset, se unifican en "Vidrio"
MAPA_CATEGORIAS = {
    "cardboard":   "Cartón",
    "brown-glass": "Vidrio",
    "green-glass": "Vidrio",
    "white-glass": "Vidrio",
    "metal":       "Metal",
    "paper":       "Papel",
    "plastic":     "Plástico",
    # battery, biological, clothes, shoes, trash se ignoran
}

# CRÍTICO: orden alfabético estricto → debe coincidir exactamente
# con el orden que asigna ImageDataGenerator (alfabético por nombre de carpeta)
CATEGORIAS = ["Cartón", "Metal", "Papel", "Plástico", "Vidrio"]

IMG_SIZE    = (224, 224)
BATCH       = 32
EPOCAS_F1   = 15   # Fase 1: solo cabeza CNN
EPOCAS_F2   = 10   # Fase 2: fine-tuning capas profundas
VAL_SPLIT   = 0.2  # 20% para validación
MODELO_PATH = "modelo_residuos.h5"
SEED        = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ═══════════════════════════════════════════════════════════
#  PASO 1 — ORGANIZAR DATASET EN train/ y val/
# ═══════════════════════════════════════════════════════════

def organizar_dataset():
    """
    Reorganiza las imágenes del dataset original en:
        dataset_listo/
            train/  Cartón/ Metal/ Papel/ Plástico/ Vidrio/
            val/    Cartón/ Metal/ Papel/ Plástico/ Vidrio/
    """
    if os.path.exists(DATASET_LISTO):
        print("[OK] Dataset ya organizado — saltando este paso.")
        return

    print("\n── Organizando dataset ──")

    # Crear estructura de carpetas
    for split in ["train", "val"]:
        for cat in CATEGORIAS:
            os.makedirs(os.path.join(DATASET_LISTO, split, cat), exist_ok=True)

    # Copiar imágenes a sus carpetas correspondientes
    for carpeta_en, cat_es in MAPA_CATEGORIAS.items():
        ruta = os.path.join(DATASET_RAW, carpeta_en)
        if not os.path.exists(ruta):
            print(f"  ⚠️  No encontrada: {ruta}")
            continue

        imagenes = [f for f in os.listdir(ruta)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        random.shuffle(imagenes)

        n_val   = int(len(imagenes) * VAL_SPLIT)
        val_set = set(imagenes[:n_val])

        for img in imagenes:
            split = "val" if img in val_set else "train"
            # Prefijo con nombre de carpeta original para evitar colisiones de nombre
            # (importante cuando varias carpetas van a la misma categoría, ej. los 3 vidrios)
            nombre_destino = f"{carpeta_en}_{img}"
            src = os.path.join(ruta, img)
            dst = os.path.join(DATASET_LISTO, split, cat_es, nombre_destino)
            shutil.copy2(src, dst)

        n_train = len(imagenes) - n_val
        print(f"  {cat_es:10s} <- {carpeta_en:15s}: {n_train} train | {n_val} val")

    print(f"\n[OK] Dataset organizado en '{DATASET_LISTO}'\n")


# ═══════════════════════════════════════════════════════════
#  PASO 2 — CONSTRUIR MODELO CNN
# ═══════════════════════════════════════════════════════════

def construir_modelo() -> Model:
    """
    Arquitectura CNN basada en MobileNetV2 (Transfer Learning):

        Input (224x224x3)
              |
        MobileNetV2 (congelado, pesos ImageNet)   <- extractor de características
              |
        GlobalAveragePooling2D
              |
        Dense(256, ReLU) -> Dropout(0.4)
              |
        Dense(128, ReLU) -> Dropout(0.3)
              |
        Dense(5, Softmax)                         <- clasificación final
    """
    base = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False  # Fase 1: base congelada, solo entrenamos la cabeza

    x      = GlobalAveragePooling2D()(base.output)
    x      = Dense(256, activation="relu")(x)
    x      = Dropout(0.4)(x)
    x      = Dense(128, activation="relu")(x)
    x      = Dropout(0.3)(x)
    salida = Dense(len(CATEGORIAS), activation="softmax")(x)

    modelo = Model(inputs=base.input, outputs=salida)
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return modelo


# ═══════════════════════════════════════════════════════════
#  PASO 3 — GENERADORES DE DATOS CON AUGMENTACIÓN
# ═══════════════════════════════════════════════════════════

def crear_generadores():
    # Augmentación para entrenamiento (genera variaciones artificiales)
    aug = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=25,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.25,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    # Validación: solo normalización, sin augmentación
    val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_ds = aug.flow_from_directory(
        os.path.join(DATASET_LISTO, "train"),
        target_size=IMG_SIZE,
        batch_size=BATCH,
        class_mode="categorical",
        seed=SEED,
    )
    val_ds = val_gen.flow_from_directory(
        os.path.join(DATASET_LISTO, "val"),
        target_size=IMG_SIZE,
        batch_size=BATCH,
        class_mode="categorical",
        seed=SEED,
    )

    print(f"Orden de clases (debe ser alfabetico): {train_ds.class_indices}")
    return train_ds, val_ds


# ═══════════════════════════════════════════════════════════
#  PASO 4 — ENTRENAMIENTO EN DOS FASES
# ═══════════════════════════════════════════════════════════

def entrenar():
    # ── Organizar dataset ──────────────────────────────────
    organizar_dataset()

    # ── Construir modelo y generadores ────────────────────
    modelo = construir_modelo()
    tr, vl = crear_generadores()

    callbacks = [
        ModelCheckpoint(
            MODELO_PATH,
            save_best_only=True,
            monitor="val_accuracy",
            verbose=1,
        ),
        EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-7, verbose=1),
    ]

    # ── FASE 1: Solo cabeza (base MobileNetV2 congelada) ──
    print("\n" + "=" * 55)
    print("  FASE 1 — Entrenando cabeza CNN")
    print("  (base MobileNetV2 congelada)")
    print("=" * 55)
    modelo.fit(tr, validation_data=vl, epochs=EPOCAS_F1, callbacks=callbacks)

    # ── FASE 2: Fine-tuning de las últimas 30 capas ───────
    print("\n" + "=" * 55)
    print("  FASE 2 — Fine-tuning (ultimas 30 capas)")
    print("  (learning rate reducido: 1e-5)")
    print("=" * 55)

    # Descongelar todas las capas y volver a congelar las primeras.
    # Se itera directamente sobre modelo.layers porque el modelo se
    # guarda aplanado (MobileNetV2 no queda como subcapa separada).
    for layer in modelo.layers:
        layer.trainable = True
    for layer in modelo.layers[:-30]:
        layer.trainable = False

    # Learning rate más bajo para no destruir los pesos pre-entrenados
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    modelo.fit(tr, validation_data=vl, epochs=EPOCAS_F2, callbacks=callbacks)

    # ── Guardar modelo y mapa de clases ───────────────────
    modelo.save(MODELO_PATH)

    mapa_path = MODELO_PATH.replace(".h5", "_clases.json")
    with open(mapa_path, "w", encoding="utf-8") as f:
        json.dump({
            "categorias": CATEGORIAS,
            "class_indices": tr.class_indices,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Modelo guardado     -> {MODELO_PATH}")
    print(f"[OK] Mapa de clases      -> {mapa_path}")
    print(f"     Orden usado         : {tr.class_indices}")
    print("\n     Ahora ejecuta: python proyectoIA.py\n")


# ═══════════════════════════════════════════════════════════
#  PUNTO DE ENTRADA
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 55)
    print("  ENTRENAMIENTO CNN — CLASIFICADOR DE RESIDUOS")
    print("  MobileNetV2 + Transfer Learning + TensorFlow")
    print("=" * 55)

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"  GPU detectada: {gpus[0].name}")
    else:
        print("  Sin GPU — usando CPU (puede tardar ~30-60 min)")

    entrenar()