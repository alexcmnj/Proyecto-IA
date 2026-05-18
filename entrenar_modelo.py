
import os, shutil, random
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

DATASET_RAW = "dataset/Garbage classification/Garbage classification"
DATASET_LISTO = "dataset_listo"
MAPA = {"cardboard":"Carton","glass":"Vidrio","metal":"Metal","paper":"Papel","plastic":"Plastico"}
CATS = ["Carton","Vidrio","Metal","Papel","Plastico"]
IMG_SIZE = (224,224)
BATCH = 32
MODELO_PATH = "modelo_residuos.h5"
random.seed(42); np.random.seed(42); tf.random.set_seed(42)

def organizar_dataset():
    if os.path.exists(DATASET_LISTO):
        print("Dataset ya organizado."); return
    print("Organizando dataset...")
    for sp in ["train","val"]:
        for c in CATS:
            os.makedirs(os.path.join(DATASET_LISTO,sp,c),exist_ok=True)
    for en,es in MAPA.items():
        folder = os.path.join(DATASET_RAW,en)
        if not os.path.exists(folder): print(f"No encontrada: {folder}"); continue
        imgs = [f for f in os.listdir(folder) if f.lower().endswith((".jpg",".jpeg",".png"))]
        random.shuffle(imgs)
        val_set = set(imgs[:int(len(imgs)*0.2)])
        for img in imgs:
            sp = "val" if img in val_set else "train"
            shutil.copy2(os.path.join(folder,img),os.path.join(DATASET_LISTO,sp,es,img))
        print(f"  {es}: {len(imgs)-len(val_set)} train | {len(val_set)} val")

def construir_modelo():
    base = MobileNetV2(input_shape=(*IMG_SIZE,3),include_top=False,weights="imagenet")
    base.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(256,activation="relu")(x); x = Dropout(0.4)(x)
    x = Dense(128,activation="relu")(x); x = Dropout(0.3)(x)
    out = Dense(len(CATS),activation="softmax")(x)
    m = Model(inputs=base.input,outputs=out)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-4),loss="categorical_crossentropy",metrics=["accuracy"])
    return m

def entrenar():
    organizar_dataset()
    modelo = construir_modelo()
    aug = ImageDataGenerator(preprocessing_function=preprocess_input,rotation_range=25,width_shift_range=0.2,height_shift_range=0.2,zoom_range=0.25,horizontal_flip=True)
    vgen = ImageDataGenerator(preprocessing_function=preprocess_input)
    tr = aug.flow_from_directory(os.path.join(DATASET_LISTO,"train"),target_size=IMG_SIZE,batch_size=BATCH,class_mode="categorical")
    vl = vgen.flow_from_directory(os.path.join(DATASET_LISTO,"val"),target_size=IMG_SIZE,batch_size=BATCH,class_mode="categorical")
    cb = [ModelCheckpoint(MODELO_PATH,save_best_only=True,monitor="val_accuracy",verbose=1),EarlyStopping(patience=5,restore_best_weights=True),ReduceLROnPlateau(factor=0.5,patience=3)]
    print("== FASE 1 =="); modelo.fit(tr,validation_data=vl,epochs=15,callbacks=cb)
    print("== FASE 2 ==")
    for layer in modelo.layers:
        layer.trainable = True
    for layer in modelo.layers[:-30]:
        layer.trainable = False
    modelo.compile(optimizer=tf.keras.optimizers.Adam(1e-5),loss="categorical_crossentropy",metrics=["accuracy"])
    modelo.fit(tr,validation_data=vl,epochs=10,callbacks=cb)
    modelo.save(MODELO_PATH); print(f"Modelo guardado: {MODELO_PATH}")

if __name__ == "__main__":
    print("="*50); print("  ENTRENAMIENTO CNN"); print("="*50)
    entrenar()
