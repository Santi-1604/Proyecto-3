
import tensorflow as tf

import mlflow.tensorflow

from get_data import *
from Indices_fin import *
from Senales_cv import *
from backtesting import *
from standarizacion_datos import *
from sklearn.utils.class_weight import compute_class_weight
w = 16
df = pd.read_csv('ORCL.csv')
df = get_data('orcl')
df = calcular_indices(df,30)

df = generar_senales(df)
df = standarizacion_datos(df)

# ===================================================
# 1. Cargar y preparar datos
# ===================================================


# Ajusta el nombre de la columna objetivo si tiene otro nombre
target_col = 'signal'
X = df.drop(columns=[target_col]).values
y = df[target_col].values

# División cronológica: 60% train, 20% test, 20% val
n = len(df)
train_end = int(0.6 * n)
test_end = int(0.8 * n)

X_train, y_train = X[:train_end], y[:train_end]
X_test, y_test = X[train_end:test_end], y[train_end:test_end]
X_val, y_val = X[test_end:], y[test_end:]

# ===================================================
# 2. Configurar MLflow
# ===================================================
mlflow.set_tracking_uri("file:///C:/Users/52331/PycharmProjects/PythonProject4/mlRUNS")
mlflow.set_experiment("MLP Experiments")
mlflow.tensorflow.autolog()
# ===================================================
# 3. balancear la funcion
# ===================================================
classes = np.unique(y_train)
y_train = np.array(y_train).astype(int)
print("Etiquetas detectadas:", classes)

class_weights_values = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)
class_weights = dict(zip(classes, class_weights_values))

# ===================================================
# 3. Función para construir modelo MLP
# ===================================================
def build_mlp(params, input_dim, output_dim):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(input_dim,)))

    # Capas ocultas
    for units in params.get("hidden_layers", [64, 32]):
        model.add(tf.keras.layers.Dense(units, activation=params.get("activation", "relu")))
        model.add(tf.keras.layers.Dropout(params.get("dropout", 0.2)))

    # Capa de salida
    model.add(tf.keras.layers.Dense(output_dim, activation=params.get("output_activation", "sigmoid")))

    # Compilación
    model.compile(
        optimizer=params.get("optimizer", "adam"),
        loss=params.get("loss", "binary_crossentropy"),
        metrics=["accuracy"]
    )

    return model

# ===================================================
# 4. Espacio de hiperparámetros
# ===================================================
param_space = [
    {"hidden_layers": [64, 32], "activation": "relu", "optimizer": "adam", "dropout": 0.2},
    {"hidden_layers": [128, 64, 32], "activation": "relu", "optimizer": "adam", "dropout": 0.3},
    {"hidden_layers": [64, 64], "activation": "tanh", "optimizer": "adam", "dropout": 0.2},
    {"hidden_layers": [128, 64,32,16], "activation": "tanh", "optimizer": "adam", "dropout": 0.1},
]

# ===================================================
# 5. Entrenamiento y registro
# ===================================================
output_dim = len(np.unique(y)) if len(np.unique(y)) > 2 else 1
loss = "sparse_categorical_crossentropy" if output_dim > 1 else "binary_crossentropy"
epochs = 50
batch_size = 32

for params in param_space:
    params["loss"] = loss
    run_name = f"MLP_{len(params['hidden_layers'])}layers_{params['activation']}"
    with mlflow.start_run(run_name=run_name):
        model = build_mlp(params, input_dim=X.shape[1], output_dim=output_dim)
        hist = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            class_weight=class_weights,
            verbose=1
        )

        val_acc = hist.history["val_accuracy"][-1]
        print(f"{run_name} -> Final Val Accuracy: {val_acc:.4f}")