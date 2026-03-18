"""
ENTRENADOR DE GESTOS - HAND2EMOJI
Entrena un Random Forest con los datos recolectados
Exporta: modelo.pkl, scaler.pkl, labels.pkl
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# CONFIGURACIÓN
# ============================================================
CSV_FILE    = 'data/mis_gestos.csv'
MODELS_DIR  = 'models'
TEST_SIZE   = 0.2     # 20% para test
RANDOM_STATE = 42
N_ESTIMATORS = 200    # Árboles en el Random Forest

# ============================================================
# EMOJI MAP
# ============================================================
EMOJI_MAP = {
    'italiano':         '🤌',
    'rock con pulgar':  '🤟',
    'rock':             '🤘',
    'corazon':          '🫶',
    'ok':               '👌',
    'pulgar':           '👍',
    'paz':              '✌️',
    'puno':             '✊',
    'llamame':          '🤙',
    'mano_abierta':     '✋',
    'indice_izquierda': '👈',
    'indice_derecha':   '👉',
    'indice_arriba':    '👆',
    'indice_abajo':     '👇',
    'dedos_cruzados':   '🤞',
    'fuck_you':         '🖕',
    'te_apunto':        '🫵',
    'pinza':            '🤏',
}


def cargar_datos():
    print("📂 Cargando datos...")
    df = pd.read_csv(CSV_FILE)
    print(f"   Total filas: {len(df)}")
    print(f"   Gestos: {df['gesto'].nunique()}")
    print(f"   Columnas: {len(df.columns)}")
    return df


def preprocesar(df):
    print("\n⚙️  Preprocesando datos...")

    # Codificar 'lado' → número (Left=0, Right=1)
    df['lado_num'] = (df['lado'] == 'Right').astype(int)

    # Separar features y etiquetas
    feature_cols = (
        [f'p{i}_x' for i in range(21)]
        + [f'p{i}_y' for i in range(21)]
        + [f'p{i}_z' for i in range(21)]
        + ['lado_num']
    )

    X = df[feature_cols].values          # (N, 64)
    y_raw = df['gesto'].values           # strings

    # Codificar etiquetas a números
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    print(f"   Features por muestra: {X.shape[1]}")
    print(f"   Clases: {list(le.classes_)}")

    return X, y, le


def dividir_y_escalar(X, y):
    print("\n✂️  Dividiendo datos (80% train / 20% test)...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y   # misma proporción de clases en train y test
    )

    print(f"   Train: {len(X_train)} muestras")
    print(f"   Test:  {len(X_test)} muestras")

    # Escalar — IMPORTANTE: fit solo en train, transform en ambos
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


def entrenar(X_train, y_train):
    print(f"\n🌲 Entrenando Random Forest ({N_ESTIMATORS} árboles)...")
    t0 = time.time()

    modelo = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,           # usa todos los núcleos del CPU
        random_state=RANDOM_STATE,
    )
    modelo.fit(X_train, y_train)

    elapsed = time.time() - t0
    print(f"   ✅ Entrenado en {elapsed:.2f} segundos")

    return modelo


def evaluar(modelo, X_test, y_test, le):
    print("\n📊 Evaluando modelo...")

    y_pred = modelo.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n   🎯 Accuracy global: {accuracy * 100:.2f}%")
    print("\n" + "=" * 65)
    print("   REPORTE POR GESTO")
    print("=" * 65)

    # Reporte con nombres de gestos
    nombres = le.classes_
    report = classification_report(y_test, y_pred, target_names=nombres)
    print(report)

    # Gestos con menor precisión
    report_dict = classification_report(
        y_test, y_pred, target_names=nombres, output_dict=True
    )
    print("=" * 65)
    print("   GESTOS QUE NECESITAN ATENCIÓN (f1-score < 0.95)")
    print("=" * 65)
    necesitan_atencion = False
    for gesto in nombres:
        f1 = report_dict[gesto]['f1-score']
        emoji = EMOJI_MAP.get(gesto, '🫱')
        if f1 < 0.95:
            print(f"   ⚠️  {emoji} {gesto}: f1={f1:.2f} → recolecta más muestras")
            necesitan_atencion = True
    if not necesitan_atencion:
        print("   ✅ Todos los gestos tienen f1-score >= 0.95 🎉")

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
    plt.title('Matriz de Confusión - Hand2Emoji', fontsize=14)
    plt.ylabel('Real')
    plt.xlabel('Predicho')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    ruta = os.path.join(MODELS_DIR, 'confusion_matrix.png')
    plt.savefig(ruta, dpi=150)
    plt.close()
    print(f"   💾 Guardada: {ruta}")


def exportar_modelos(modelo, scaler, le):
    print("\n💾 Exportando modelos...")
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Guardar modelo
    ruta_modelo = os.path.join(MODELS_DIR, 'modelo.pkl')
    with open(ruta_modelo, 'wb') as f:
        pickle.dump(modelo, f)
    print(f"   ✅ {ruta_modelo}")

    # Guardar scaler (CRÍTICO: debe acompañar siempre al modelo)
    ruta_scaler = os.path.join(MODELS_DIR, 'scaler.pkl')
    with open(ruta_scaler, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"   ✅ {ruta_scaler}")

    # Guardar LabelEncoder (mapeo número → nombre gesto)
    ruta_labels = os.path.join(MODELS_DIR, 'labels.pkl')
    with open(ruta_labels, 'wb') as f:
        pickle.dump(le, f)
    print(f"   ✅ {ruta_labels}")

    # Guardar metadata útil
    metadata = {
        'gestos': list(le.classes_),
        'emoji_map': EMOJI_MAP,
        'n_features': 64,
        'n_clases': len(le.classes_),
        'n_estimators': N_ESTIMATORS,
    }
    ruta_meta = os.path.join(MODELS_DIR, 'metadata.pkl')
    with open(ruta_meta, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"   ✅ {ruta_meta}")

    # Tamaños de archivos
    print("\n   📦 Tamaños:")
    for ruta in [ruta_modelo, ruta_scaler, ruta_labels, ruta_meta]:
        size_kb = os.path.getsize(ruta) / 1024
        print(f"      {os.path.basename(ruta)}: {size_kb:.1f} KB")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 65)
    print("🤖 ENTRENADOR HAND2EMOJI")
    print("=" * 65)

    try:
        # Pipeline completo
        df                                    = cargar_datos()
        X, y, le                              = preprocesar(df)
        X_train, X_test, y_train, y_test, sc  = dividir_y_escalar(X, y)
        modelo                                = entrenar(X_train, y_train)
        y_pred, accuracy                      = evaluar(modelo, X_test, y_test, le)
        exportar_modelos(modelo, sc, le)
        guardar_matriz_confusion(y_test, y_pred, le)

        print("\n" + "=" * 65)
        if accuracy >= 0.95:
            print(f"🎉 ¡Modelo listo! Accuracy: {accuracy*100:.2f}%")
            print("🚀 Siguiente paso: ejecuta detector.py")
        else:
            print(f"⚠️  Accuracy: {accuracy*100:.2f}% — considera recolectar")
            print("   más muestras de los gestos marcados con ⚠️")
        print("=" * 65)

    except FileNotFoundError:
        print(f"\n❌ No se encontró {CSV_FILE}")
        print("   Asegúrate de haber ejecutado recolector.py primero")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        raise