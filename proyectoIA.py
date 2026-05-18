"""
=============================================================
  CLASIFICADOR DE RESIDUOS CON CNN
  Tkinter + TensorFlow/Keras (MobileNetV2)
  Cámara en tiempo real + carga de imágenes
=============================================================
  Categorías: Papel, Plástico, Metal, Vidrio, Cartón
  Modelo: MobileNetV2 (Transfer Learning)
  Requisito: Python 3.9+
=============================================================
"""

import os
import io
import time
import threading
import tkinter as tk
from tkinter import filedialog, ttk
from datetime import datetime
from collections import defaultdict

import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont

# ── TensorFlow / Keras ──────────────────────────────────────
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── OpenCV (cámara) ─────────────────────────────────────────
import cv2


# ═══════════════════════════════════════════════════════════
#  CONFIGURACIÓN GLOBAL
# ═══════════════════════════════════════════════════════════
CATEGORIAS   = ["Cartón", "Metal", "Papel", "Plástico", "Vidrio"]
IMG_SIZE     = (224, 224)
MODELO_PATH  = "modelo_residuos.h5"
CONF_UMBRAL  = 0.30   # confianza mínima para reportar detección
CAM_FPS      = 15     # inferencia cada N ms
COLORES_CAT  = {
    "Papel":    "#F5A623",
    "Plástico": "#2196F3",
    "Metal":    "#43A047",
    "Vidrio":   "#9C27B0",
    "Cartón":   "#795548",
}
COLOR_FONDO  = "#EAF3F8"
COLOR_PANEL  = "#FFFFFF"
COLOR_BORDE  = "#D0DDE8"
COLOR_TITULO = "#1A2E44"
COLOR_MUTED  = "#5A7A9A"


# ═══════════════════════════════════════════════════════════
#  CNN — CONSTRUCCIÓN DEL MODELO
# ═══════════════════════════════════════════════════════════
def construir_modelo(num_clases: int = 5) -> Model:
    """
    Arquitectura CNN basada en MobileNetV2:

        Input (224×224×3)
              ↓
        MobileNetV2 (congelado, ImageNet)   ← extractor de características
              ↓
        GlobalAveragePooling2D
              ↓
        Dense(256, ReLU) → Dropout(0.4)
              ↓
        Dense(128, ReLU) → Dropout(0.3)
              ↓
        Dense(num_clases, Softmax)          ← clasificación final
    """
    base = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False  # Transfer Learning: solo entrenamos la cabeza

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    salida = Dense(num_clases, activation="softmax")(x)

    modelo = Model(inputs=base.input, outputs=salida)
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return modelo


def cargar_o_crear_modelo() -> Model:
    if os.path.exists(MODELO_PATH):
        print(f"[CNN] Cargando modelo entrenado: {MODELO_PATH}")
        return tf.keras.models.load_model(MODELO_PATH)
    print("[CNN] Creando arquitectura base (sin entrenamiento previo)")
    return construir_modelo(len(CATEGORIAS))


# ═══════════════════════════════════════════════════════════
#  CNN — PREPROCESAMIENTO E INFERENCIA
# ═══════════════════════════════════════════════════════════
def preprocesar(pil_img: Image.Image) -> np.ndarray:
    """Preprocesa imagen PIL para MobileNetV2."""
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)            # normalización MobileNetV2
    return np.expand_dims(arr, axis=0)    # (1, 224, 224, 3)


def clasificar(modelo: Model, pil_img: Image.Image) -> dict:
    """
    Ejecuta la CNN sobre la imagen y retorna:
        label       : categoría predicha
        confianza   : porcentaje (0-100)
        top3        : lista de las 3 mejores predicciones
        tiempo_ms   : tiempo de inferencia en milisegundos
    """
    t0    = time.perf_counter()
    entrada = preprocesar(pil_img)
    preds   = modelo.predict(entrada, verbose=0)[0]  # array (5,)
    tiempo  = int((time.perf_counter() - t0) * 1000)

    idx_max    = int(np.argmax(preds))
    confianza  = float(preds[idx_max])

    top3 = sorted(
        [(CATEGORIAS[i], float(preds[i])) for i in range(len(CATEGORIAS))],
        key=lambda x: -x[1],
    )[:3]

    no_encontrado = confianza < CONF_UMBRAL
    return {
        "label":      "No encontrado" if no_encontrado else CATEGORIAS[idx_max],
        "confianza":  round(confianza * 100, 1),
        "detectado":  not no_encontrado,
        "top3":       top3,
        "tiempo_ms":  tiempo,
    }


# ═══════════════════════════════════════════════════════════
#  CNN — ENTRENAMIENTO (llamar manualmente si tienes dataset)
# ═══════════════════════════════════════════════════════════
def entrenar(data_dir: str = "dataset", epocas: int = 20, batch: int = 32):
    """
    Entrena el modelo en dos fases con datos propios.

    Estructura esperada en data_dir:
        dataset/
            train/  Papel/ Plástico/ Metal/ Vidrio/ Cartón/
            val/    Papel/ Plástico/ Metal/ Vidrio/ Cartón/

    Datasets sugeridos:
        - Kaggle: Waste Classification Data (Sashaank Sekar)
        - TrashNet (Gary Thung & Mindy Yang)
    """
    modelo = construir_modelo(len(CATEGORIAS))

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
    val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_ds = aug.flow_from_directory(
        f"{data_dir}/train", target_size=IMG_SIZE,
        batch_size=batch, class_mode="categorical",
    )
    val_ds = val_gen.flow_from_directory(
        f"{data_dir}/val", target_size=IMG_SIZE,
        batch_size=batch, class_mode="categorical",
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            MODELO_PATH, save_best_only=True, monitor="val_accuracy"
        ),
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-7),
    ]

    # Fase 1: Solo cabeza (base congelada)
    print("── Fase 1: entrenando cabeza CNN ──")
    modelo.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks)

    # Fase 2: Fine-tuning últimas 30 capas de MobileNetV2
    print("── Fase 2: fine-tuning ──")
    modelo.layers[0].trainable = True
    for capa in modelo.layers[0].layers[:-30]:
        capa.trainable = False
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    modelo.fit(train_ds, validation_data=val_ds, epochs=epocas, callbacks=callbacks)
    modelo.save(MODELO_PATH)
    print(f"[OK] Modelo guardado → {MODELO_PATH}")


# ═══════════════════════════════════════════════════════════
#  DIBUJAR ANOTACIONES SOBRE IMAGEN PIL
# ═══════════════════════════════════════════════════════════
def anotar_imagen(pil_img: Image.Image, resultado: dict) -> Image.Image:
    """Dibuja la caja de predicción y etiqueta sobre la imagen."""
    img  = pil_img.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    W, H = img.size
    color = COLORES_CAT.get(resultado["label"], "#607D8B")

    # Caja bounding (toda la imagen — se puede adaptar a YOLO)
    pad = 10
    draw.rectangle([pad, pad, W - pad, H - pad], outline=color, width=3)

    # Etiqueta de fondo
    texto = f"{resultado['label']}  {resultado['confianza']}%"
    try:
        fuente = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        fuente = ImageFont.load_default()

    bbox_txt = draw.textbbox((pad, pad), texto, font=fuente)
    draw.rectangle(
        [bbox_txt[0] - 4, bbox_txt[1] - 4, bbox_txt[2] + 6, bbox_txt[3] + 4],
        fill=color,
    )
    draw.text((pad, pad), texto, fill="white", font=fuente)

    return img


# ═══════════════════════════════════════════════════════════
#  INTERFAZ GRÁFICA — TKINTER
# ═══════════════════════════════════════════════════════════
class AppClasificador(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Clasificador de Residuos — CNN")
        self.configure(bg=COLOR_FONDO)
        self.resizable(True, True)
        self.minsize(900, 600)

        # Responsive: ocupa 85% de la pantalla
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w, h   = min(1100, sw - 40), min(700, sh - 60)
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

        # Estado
        self.modelo     = None
        self.camara     = None
        self.corriendo  = False
        self.contadores = defaultdict(int)
        self.logs       = []
        self._img_tk    = None   # referencia para evitar GC

        self._cargar_modelo_async()
        self._construir_ui()

    # ─── Carga del modelo en hilo separado ───────────────
    def _cargar_modelo_async(self):
        def _cargar():
            self.modelo = cargar_o_crear_modelo()
            self.after(0, lambda: self._set_estado("Modelo CNN listo ✓", "green"))
        threading.Thread(target=_cargar, daemon=True).start()

    # ─── Construcción de la UI ───────────────────────────
    def _construir_ui(self):
        # ── Encabezado ───────────────────────────────────
        enc = tk.Frame(self, bg=COLOR_FONDO)
        enc.pack(fill="x", padx=16, pady=(14, 6))

        icono_frame = tk.Frame(enc, bg="#F5A623", width=40, height=40)
        icono_frame.pack(side="left")
        icono_frame.pack_propagate(False)
        tk.Label(icono_frame, text="📷", bg="#F5A623", font=("Segoe UI", 18)).pack(expand=True)

        info = tk.Frame(enc, bg=COLOR_FONDO)
        info.pack(side="left", padx=10)
        tk.Label(info, text="Clasificador de Residuos — CNN",
                 bg=COLOR_FONDO, fg=COLOR_TITULO,
                 font=("Segoe UI", 16, "bold")).pack(anchor="w")
        tk.Label(info, text="MobileNetV2 · Transfer Learning · TensorFlow/Keras",
                 bg=COLOR_FONDO, fg=COLOR_MUTED,
                 font=("Segoe UI", 10)).pack(anchor="w")

        self.lbl_estado = tk.Label(enc, text="⏳ Cargando modelo CNN...",
                                   bg=COLOR_FONDO, fg="#F5A623",
                                   font=("Segoe UI", 10, "bold"))
        self.lbl_estado.pack(side="right", padx=6)

        # ── Layout principal ──────────────────────────────
        main = tk.Frame(self, bg=COLOR_FONDO)
        main.pack(fill="both", expand=True, padx=16, pady=6)

        # Columna izquierda — video/imagen
        self.panel_izq = tk.Frame(main, bg=COLOR_FONDO)
        self.panel_izq.pack(side="left", fill="both", expand=True)

        # Columna derecha — estadísticas
        self.panel_der = tk.Frame(main, bg=COLOR_FONDO, width=280)
        self.panel_der.pack(side="right", fill="y", padx=(10, 0))
        self.panel_der.pack_propagate(False)

        self._ui_video()
        self._ui_controles()
        self._ui_estadisticas()

    # ─── Panel de video / imagen ─────────────────────────
    def _ui_video(self):
        card = self._card(self.panel_izq)
        card.pack(fill="both", expand=True)

        self._lbl_video_title = tk.Label(card, text="📹  Video en Tiempo Real",
                                         bg=COLOR_PANEL, fg=COLOR_MUTED,
                                         font=("Segoe UI", 10, "bold"))
        self._lbl_video_title.pack(anchor="w", padx=4, pady=(0, 6))

        self.canvas_video = tk.Label(
            card, bg="#1A1A2E", text="",
            relief="flat", cursor="crosshair"
        )
        self.canvas_video.pack(fill="both", expand=True)

        # Texto placeholder
        self._placeholder()

    def _placeholder(self):
        ph = Image.new("RGB", (560, 360), "#1A2E44")
        draw = ImageDraw.Draw(ph)
        try:
            f = ImageFont.truetype("arial.ttf", 15)
        except Exception:
            f = ImageFont.load_default()
        msg = "Presiona 'Cámara' o 'Cargar imagen'"
        draw.text((280, 180), msg, fill="#5A7A9A", font=f, anchor="mm")
        self._mostrar_frame(ph)

    # ─── Controles ───────────────────────────────────────
    def _ui_controles(self):
        frm = tk.Frame(self.panel_izq, bg=COLOR_FONDO)
        frm.pack(fill="x", pady=8)

        self._btn(frm, "📷  Cámara",       "#2196F3", self.iniciar_camara).pack(side="left", padx=4)
        self._btn(frm, "🖼  Cargar imagen", "#43A047", self.cargar_imagen).pack(side="left", padx=4)
        self._btn(frm, "⏹  Detener",       "#E53935", self.detener).pack(side="left", padx=4)
        self._btn(frm, "🔄  Reiniciar",     "#757575", self.reiniciar).pack(side="left", padx=4)

        # Barra de estado inferior
        self.lbl_fps = tk.Label(self.panel_izq,
                                text="Tiempo por frame: — ms",
                                bg=COLOR_FONDO, fg=COLOR_MUTED,
                                font=("Segoe UI", 9))
        self.lbl_fps.pack(anchor="w")

    # ─── Estadísticas (panel derecho) ────────────────────
    def _ui_estadisticas(self):
        # Confianza
        c1 = self._card(self.panel_der)
        c1.pack(fill="x", pady=(0, 8))
        tk.Label(c1, text="CONFIANZA", bg=COLOR_PANEL, fg=COLOR_MUTED,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self.barra_conf = ttk.Progressbar(c1, length=240, mode="determinate",
                                          maximum=100)
        self.barra_conf.pack(fill="x", pady=4)
        self.lbl_conf = tk.Label(c1, text="—", bg=COLOR_PANEL, fg=COLOR_TITULO,
                                 font=("Segoe UI", 14, "bold"))
        self.lbl_conf.pack(anchor="w")

        # Tiempo por frame
        c2 = self._card(self.panel_der)
        c2.pack(fill="x", pady=(0, 8))
        tk.Label(c2, text="TIEMPO POR FRAME", bg=COLOR_PANEL, fg=COLOR_MUTED,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self.lbl_tiempo = tk.Label(c2, text="— ms", bg=COLOR_PANEL,
                                   fg=COLOR_TITULO, font=("Segoe UI", 24, "bold"))
        self.lbl_tiempo.pack(anchor="w")

        # Contadores por categoría
        c3 = self._card(self.panel_der)
        c3.pack(fill="x", pady=(0, 8))
        tk.Label(c3, text="CONTADOR POR CATEGORÍA (TURNO)",
                 bg=COLOR_PANEL, fg=COLOR_MUTED,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(0, 6))

        self.lbls_cont = {}
        emojis = {"Papel": "📄", "Plástico": "🧴", "Metal": "🥫",
                  "Vidrio": "🍶", "Cartón": "📦"}
        for cat in CATEGORIAS:
            fila = tk.Frame(c3, bg=COLORES_CAT[cat], height=36,
                            cursor="hand2")
            fila.pack(fill="x", pady=2)
            fila.pack_propagate(False)
            tk.Label(fila, text=f"  {emojis[cat]}  {cat}",
                     bg=COLORES_CAT[cat], fg="white",
                     font=("Segoe UI", 11, "bold")).pack(side="left")
            lbl = tk.Label(fila, text="0", bg=COLORES_CAT[cat], fg="white",
                           font=("Segoe UI", 14, "bold"), padx=10)
            lbl.pack(side="right")
            self.lbls_cont[cat] = lbl
            # Clic para incrementar manualmente
            fila.bind("<Button-1>", lambda e, c=cat: self._incrementar(c))

        # Top-3 predicciones
        c4 = self._card(self.panel_der)
        c4.pack(fill="x", pady=(0, 8))
        tk.Label(c4, text="TOP PREDICCIONES CNN", bg=COLOR_PANEL, fg=COLOR_MUTED,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(0, 4))
        self.lbl_top3 = tk.Label(c4, text="Sin detecciones", bg=COLOR_PANEL,
                                 fg=COLOR_MUTED, font=("Segoe UI", 10),
                                 justify="left")
        self.lbl_top3.pack(anchor="w")

        # Log de detecciones
        c5 = self._card(self.panel_der)
        c5.pack(fill="both", expand=True)
        tk.Label(c5, text="LOG DE DETECCIONES", bg=COLOR_PANEL, fg=COLOR_MUTED,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(0, 4))
        self.txt_log = tk.Text(c5, height=6, bg="#F8FAFC", fg=COLOR_MUTED,
                               font=("Courier", 9), state="disabled",
                               relief="flat", wrap="word")
        self.txt_log.pack(fill="both", expand=True)
        scroll = ttk.Scrollbar(c5, command=self.txt_log.yview)
        self.txt_log["yscrollcommand"] = scroll.set

    # ─── Cámara en tiempo real ───────────────────────────
    def iniciar_camara(self):
        if self.corriendo:
            return
        self.detener()
        self.camara = cv2.VideoCapture(0)
        if not self.camara.isOpened():
            self._set_estado("❌ No se encontró cámara", "red")
            return
        self.corriendo = True
        self._set_estado("🟢  Cámara activa · Clasificando...", "green")
        self._lbl_video_title.config(text="📹  Video en Tiempo Real")
        self._loop_camara()

    def _loop_camara(self):
        if not self.corriendo or self.camara is None:
            return
        ret, frame = self.camara.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img   = Image.fromarray(frame_rgb)

            if self.modelo is not None:
                resultado = clasificar(self.modelo, pil_img)
                if resultado["detectado"]:
                    pil_img = anotar_imagen(pil_img, resultado)
                    self._actualizar_stats(resultado)
                self.lbl_fps.config(
                    text=f"Tiempo por frame (CNN): {resultado['tiempo_ms']} ms"
                )
            self._mostrar_frame(pil_img)

        self.after(1000 // CAM_FPS, self._loop_camara)

    # ─── Cargar imagen desde disco ───────────────────────
    def cargar_imagen(self):
        self.detener()
        path = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.webp"), ("Todos", "*.*")]
        )
        if not path:
            return
        self._set_estado("⏳  Clasificando con CNN...", "#F5A623")
        self._lbl_video_title.config(text="🖼  Imagen cargada")

        def _procesar():
            pil_img  = Image.open(path).convert("RGB")
            resultado = clasificar(self.modelo, pil_img)
            anotada   = anotar_imagen(pil_img, resultado)
            self.after(0, lambda: self._mostrar_frame(anotada))
            self.after(0, lambda: self._actualizar_stats(resultado))
            self.after(0, lambda: self._set_estado(
                f"✅  {resultado['label']} — {resultado['confianza']}%  ({resultado['tiempo_ms']} ms)",
                "green"
            ))
            self.after(0, lambda: self.lbl_fps.config(
                text=f"Tiempo de inferencia CNN: {resultado['tiempo_ms']} ms"
            ))

        threading.Thread(target=_procesar, daemon=True).start()

    # ─── Actualizar widgets de estadísticas ─────────────
    def _actualizar_stats(self, resultado: dict):
        cat  = resultado["label"]
        conf = resultado["confianza"]
        ms   = resultado["tiempo_ms"]

        self.barra_conf["value"] = conf
        self.lbl_conf.config(text=f"{conf}%")
        self.lbl_tiempo.config(text=f"{ms} ms")

        self.contadores[cat] += 1
        self.lbls_cont[cat].config(text=str(self.contadores[cat]))

        # Top-3
        top3_txt = "\n".join(
            f"  {'▶' if i==0 else '  '} {c}:  {p*100:.1f}%"
            for i, (c, p) in enumerate(resultado["top3"])
        )
        self.lbl_top3.config(text=top3_txt)

        # Log
        hora = datetime.now().strftime("%H:%M:%S")
        linea = f"{hora}  [{cat}]  {conf}%\n"
        self.txt_log.config(state="normal")
        self.txt_log.insert("end", linea)
        self.txt_log.see("end")
        self.txt_log.config(state="disabled")

    # ─── Mostrar frame en canvas ─────────────────────────
    def _mostrar_frame(self, pil_img: Image.Image):
        # Escalar al tamaño real del widget
        self.canvas_video.update_idletasks()
        w = max(self.canvas_video.winfo_width(), 400)
        h = max(self.canvas_video.winfo_height(), 300)
        pil_img.thumbnail((w, h), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(pil_img)
        self.canvas_video.config(image=img_tk)
        self._img_tk = img_tk  # evitar garbage collection

    # ─── Helpers ─────────────────────────────────────────
    def detener(self):
        self.corriendo = False
        if self.camara:
            self.camara.release()
            self.camara = None
        self._set_estado("⏹  Detenido", COLOR_MUTED)

    def reiniciar(self):
        self.detener()
        for cat in CATEGORIAS:
            self.contadores[cat] = 0
            self.lbls_cont[cat].config(text="0")
        self.txt_log.config(state="normal")
        self.txt_log.delete("1.0", "end")
        self.txt_log.config(state="disabled")
        self.barra_conf["value"] = 0
        self.lbl_conf.config(text="—")
        self.lbl_tiempo.config(text="— ms")
        self.lbl_top3.config(text="Sin detecciones")
        self._placeholder()
        self._set_estado("Reiniciado", COLOR_MUTED)

    def _incrementar(self, cat: str):
        self.contadores[cat] += 1
        self.lbls_cont[cat].config(text=str(self.contadores[cat]))

    def _set_estado(self, texto: str, color: str = COLOR_MUTED):
        self.lbl_estado.config(text=texto, fg=color)

    def _card(self, parent) -> tk.Frame:
        frm = tk.Frame(parent, bg=COLOR_PANEL,
                       highlightbackground=COLOR_BORDE,
                       highlightthickness=1,
                       padx=12, pady=10)
        return frm

    def _btn(self, parent, texto: str, color: str, cmd) -> tk.Button:
        return tk.Button(
            parent, text=texto, command=cmd,
            bg=color, fg="white", activebackground=color,
            font=("Segoe UI", 10, "bold"),
            relief="flat", padx=12, pady=6, cursor="hand2",
        )

    def on_close(self):
        self.detener()
        self.destroy()


# ═══════════════════════════════════════════════════════════
#  PUNTO DE ENTRADA
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 55)
    print("  CLASIFICADOR DE RESIDUOS CON CNN")
    print("  MobileNetV2 + Transfer Learning + TensorFlow")
    print("=" * 55)

    # Para entrenar con tu propio dataset descomenta:
    # entrenar(data_dir="dataset", epocas=20, batch=32)

    app = AppClasificador()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()