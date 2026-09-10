"""
ENTRENADOR DE GESTOS - HAND2EMOJI (FASE 2)
- Normalización canónica (espejado para mano izquierda) -> 63 features invariantes a la mano
- Data Augmentation sintético (jittering, escala, rotación)
- Modelo ultra-liviano MLPClassifier (< 50 KB)
- Exporta: modelo.onnx, labels.json, modelo.pkl, scaler.pkl, labels.pkl, metadata.pkl
"""

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import pickle
import json
import shutil
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
import gestures_common as gc

# ============================================================
# CONFIGURACIÓN
# ============================================================
CSV_FILE           = 'data/mis_gestos.csv'
MODELS_DIR         = 'models'
DOCS_MODELS_DIR    = 'docs/models'
TEST_SIZE          = 0.2     # 20% para test
RANDOM_STATE       = 42
AUGMENTATION_FACTOR = 2       # Duplicar muestras de entrenamiento con variaciones

EMOJI_MAP = gc.EMOJI_MAP


def cargar_datos():
    print("📂 Cargando datos...")
    df = pd.read_csv(CSV_FILE)
    print(f"   Total filas: {len(df)}")
    print(f"   Gestos: {df['gesto'].nunique()}")
    print(f"   Columnas: {len(df.columns)}")
    return df


def preprocesar(df):
    """
    Preprocesa los datos aplicando normalización canónica:
    Para muestras con lado == 'Left', invierte las coordenadas X (primeras 21 columnas)
    para unificarlas en el sistema de coordenadas de la mano derecha.
    Retorna X (N, 63), y codificado y el LabelEncoder.
    """
    print("\n⚙️  Preprocesando datos con espejado canónico...")

    # Usar las 63 columnas de landmarks canónicos
    feature_cols = gc.obtener_columnas_canonicas()
    X = df[feature_cols].values.copy().astype(np.float32)   # (N, 63)

    # Espejado canónico: invertir eje X si la mano es izquierda
    es_izq = (df['lado'] == 'Left').values
    X[es_izq, :gc.NUM_LANDMARKS] = -X[es_izq, :gc.NUM_LANDMARKS]
    print(f"   Muestras de mano izquierda espejadas a canónicas: {np.sum(es_izq)}")

    y_raw = df['gesto'].values
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    print(f"   Features por muestra: {X.shape[1]} (canónicos)")
    print(f"   Clases ({len(le.classes_)}): {list(le.classes_)}")

    return X, y, le


def aumentar_datos(X: np.ndarray, y: np.ndarray, factor: int = 2, random_state: int = 42) -> tuple:
    """
    Genera variaciones sintéticas realistas para landmarks 3D (63 features):
    - Jittering gaussiano (ruido leve en sensores)
    - Variación de escala (+/- 7%)
    - Rotación leve en el plano XY (+/- 7 grados)
    """
    print(f"\n🔄 Aplicando Data Augmentation (factor={factor})...")
    rng = np.random.default_rng(random_state)
    X_aumentado = [X]
    y_aumentado = [y]

    num_muestras = X.shape[0]

    for f in range(factor):
        X_nuevo = X.copy()

        # 1. Jittering gaussiano (desviación 0.012)
        ruido = rng.normal(0, 0.012, X.shape).astype(np.float32)
        X_nuevo += ruido

        # 2. Escala aleatoria entre 0.93 y 1.07
        factor_escala = rng.uniform(0.93, 1.07, size=(num_muestras, 1)).astype(np.float32)
        X_nuevo *= factor_escala

        # 3. Rotación leve en el plano XY (+/- 7 grados)
        angulos_deg = rng.uniform(-7.0, 7.0, size=num_muestras)
        angulos_rad = np.radians(angulos_deg)
        cos_a = np.cos(angulos_rad).astype(np.float32)
        sin_a = np.sin(angulos_rad).astype(np.float32)

        for i in range(gc.NUM_LANDMARKS):
            xi = X_nuevo[:, i].copy()
            yi = X_nuevo[:, gc.NUM_LANDMARKS + i].copy()
            X_nuevo[:, i] = xi * cos_a - yi * sin_a
            X_nuevo[:, gc.NUM_LANDMARKS + i] = xi * sin_a + yi * cos_a

        X_aumentado.append(X_nuevo)
        y_aumentado.append(y)

    X_total = np.vstack(X_aumentado)
    y_total = np.concatenate(y_aumentado)
    print(f"   Muestras de train aumentadas: {len(X)} -> {len(X_total)}")
    return X_total, y_total


def dividir_y_escalar(X, y):
    print("\n✂️  Dividiendo datos (80% train / 20% test)...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"   Train original: {len(X_train)} muestras")
    print(f"   Test (sin alterar): {len(X_test)} muestras")

    # Aumentar únicamente el conjunto de entrenamiento (test se mantiene intacto)
    X_train_aug, y_train_aug = aumentar_datos(X_train, y_train, factor=AUGMENTATION_FACTOR)

    # Escalar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_aug)
    X_test_scaled  = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train_aug, y_test, scaler, X_test


def entrenar(X_train, y_train):
    print("\n🧠 Entrenando MLPClassifier (Red Neuronal Liviana)...")
    t0 = time.time()

    modelo = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        alpha=1e-4,
        batch_size=64,
        learning_rate_init=1e-3,
        max_iter=400,
        early_stopping=True,
        n_iter_no_change=25,
        random_state=RANDOM_STATE,
    )
    modelo.fit(X_train, y_train)

    elapsed = time.time() - t0
    print(f"   ✅ Entrenado en {elapsed:.2f} segundos ({modelo.n_iter_} iteraciones)")

    return modelo


def evaluar(modelo, X_test, y_test, le):
    print("\n📊 Evaluando modelo...")

    y_pred = modelo.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n   🎯 Accuracy global: {accuracy * 100:.2f}%")
    print("\n" + "=" * 65)
    print("   REPORTE POR GESTO")
    print("=" * 65)

    nombres = le.classes_
    report = classification_report(y_test, y_pred, target_names=nombres)
    print(report)

    return y_pred, accuracy


def guardar_matriz_confusion(y_test, y_pred, le):
    print("\n📈 Generando matriz de confusión...")

    nombres = le.classes_
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=nombres,
        yticklabels=nombres,
    )
    plt.title('Matriz de Confusión - Hand2Emoji (Fase 2)', fontsize=14)
    plt.ylabel('Real')
    plt.xlabel('Predicho')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    os.makedirs(MODELS_DIR, exist_ok=True)
    ruta = os.path.join(MODELS_DIR, 'confusion_matrix.png')
    plt.savefig(ruta, dpi=150)
    plt.close()
    print(f"   💾 Guardada: {ruta}")


def exportar_modelos(modelo, scaler, le):
    print("\n💾 Exportando modelos...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DOCS_MODELS_DIR, exist_ok=True)

    # 1. Guardar modelo pickle
    ruta_modelo = os.path.join(MODELS_DIR, 'modelo.pkl')
    with open(ruta_modelo, 'wb') as f:
        pickle.dump(modelo, f)
    print(f"   ✅ {ruta_modelo}")

    # 2. Guardar scaler
    ruta_scaler = os.path.join(MODELS_DIR, 'scaler.pkl')
    with open(ruta_scaler, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"   ✅ {ruta_scaler}")

    # 3. Guardar LabelEncoder
    ruta_labels = os.path.join(MODELS_DIR, 'labels.pkl')
    with open(ruta_labels, 'wb') as f:
        pickle.dump(le, f)
    print(f"   ✅ {ruta_labels}")

    # 4. Guardar metadata útil
    metadata = {
        'gestos': list(le.classes_),
        'emoji_map': {cls: EMOJI_MAP.get(cls, '🫱') for cls in le.classes_},
        'n_features': 63,
        'n_clases': len(le.classes_),
        'modelo_tipo': 'MLPClassifier',
        'canonical_mirroring': True,
    }
    ruta_meta = os.path.join(MODELS_DIR, 'metadata.pkl')
    with open(ruta_meta, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"   ✅ {ruta_meta}")

    # 5. Exportar a formato ONNX (Pipeline con Scaler + Modelo integrados, sin zipmap para JS)
    pipeline_completo = Pipeline([
        ('scaler', scaler),
        ('classifier', modelo)
    ])
    tipo_entrada = [('float_input', FloatTensorType([None, 63]))]
    modelo_onnx = convert_sklearn(
        pipeline_completo,
        initial_types=tipo_entrada,
        target_opset=12,
        options={id(modelo): {'zipmap': False}}
    )

    ruta_onnx = os.path.join(MODELS_DIR, 'modelo.onnx')
    with open(ruta_onnx, 'wb') as f:
        f.write(modelo_onnx.SerializeToString())
    print(f"   ✅ ONNX: {ruta_onnx} ({os.path.getsize(ruta_onnx)/1024:.1f} KB)")

    # 6. Exportar labels.json para la web
    metadata_json = {
        "n_features": 63,
        "classes": list(le.classes_),
        "emoji_map": {cls: EMOJI_MAP.get(cls, '🫱') for cls in le.classes_},
        "canonical_mirroring": True
    }
    ruta_json = os.path.join(MODELS_DIR, 'labels.json')
    with open(ruta_json, 'w', encoding='utf-8') as f:
        json.dump(metadata_json, f, ensure_ascii=False, indent=2)
    print(f"   ✅ JSON: {ruta_json}")

    # 7. Copiar modelo.onnx y labels.json a docs/models/ para consumo directo en GitHub Pages
    shutil.copy2(ruta_onnx, os.path.join(DOCS_MODELS_DIR, 'modelo.onnx'))
    shutil.copy2(ruta_json, os.path.join(DOCS_MODELS_DIR, 'labels.json'))
    print(f"   ✅ Copiado a {DOCS_MODELS_DIR} para GitHub Pages")

    # Resumen de tamaños
    print("\n   📦 Resumen de tamaños:")
    for ruta in [ruta_onnx, ruta_json, ruta_modelo, ruta_scaler]:
        size_kb = os.path.getsize(ruta) / 1024
        print(f"      {os.path.basename(ruta)}: {size_kb:.1f} KB")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 65)
    print("🤖 ENTRENADOR HAND2EMOJI (FASE 2 - CANÓNICO & ONNX)")
    print("=" * 65)

    try:
        df                                             = cargar_datos()
        X, y, le                                       = preprocesar(df)
        X_train, X_test, y_train, y_test, sc, X_te_raw = dividir_y_escalar(X, y)
        modelo                                         = entrenar(X_train, y_train)
        y_pred, accuracy                               = evaluar(modelo, X_test, y_test, le)
        exportar_modelos(modelo, sc, le)
        guardar_matriz_confusion(y_test, y_pred, le)

        print("\n" + "=" * 65)
        print(f"🎉 ¡Modelo Fase 2 completado! Accuracy: {accuracy*100:.2f}%")
        print("=" * 65)

    except FileNotFoundError:
        print(f"\n❌ No se encontró {CSV_FILE}")
        print("   Asegúrate de haber ejecutado recolector.py primero")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        raise