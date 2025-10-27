
import tensorflow as tf

import mlflow.tensorflow
from sklearn.utils.class_weight import compute_class_weight

from get_data import *
from Indices_fin import *
from Senales_cv import *
from backtesting import *
from standarizacion_datos import *
w = 16

df = get_data('orcl')
df = calcular_indices(df,30)

df = generar_senales(df)
df = standarizacion_datos(df)


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

# CNN requiere una dimensión adicional (tipo canal)
# Conv1D espera entrada (samples, timesteps, features)
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)
X_val = np.expand_dims(X_val, axis=2)

# ===================================================
# 2. Configurar MLflow
# ===================================================
mlflow.set_tracking_uri("file:///C:/Users/52331/PycharmProjects/PythonProject4/mlRUNS")
mlflow.set_experiment("CNN Experiments")
mlflow.tensorflow.autolog()

# ===================================================
# 3. Calcular class weights (opcional si hay sesgo)
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
# 4. Función para construir CNN
# ===================================================
def build_cnn(params, input_shape, output_dim):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))

    filters = params.get("filters", 32)
    activation = params.get("activation", "relu")
    conv_layers = params.get("conv_layers", 2)

    # Capas convolucionales
    for _ in range(conv_layers):
        model.add(tf.keras.layers.Conv1D(filters, kernel_size=3, activation=activation, padding="same"))
        model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
        model.add(tf.keras.layers.Dropout(params.get("dropout", 0.2)))
        filters *= 2  # duplicar filtros por capa para captar patrones más complejos

    # Capas densas finales
    model.add(tf.keras.layers.Flatten())
    for units in params.get("dense_layers", [64]):
        model.add(tf.keras.layers.Dense(units, activation=activation))
        model.add(tf.keras.layers.Dropout(params.get("dropout_dense", 0.2)))

    # Capa de salida
    output_activation = "sigmoid" if output_dim == 1 else "softmax"
    loss = "binary_crossentropy" if output_dim == 1 else "sparse_categorical_crossentropy"

    model.add(tf.keras.layers.Dense(output_dim, activation=output_activation))
    model.compile(
        optimizer=params.get("optimizer", "adam"),
        loss=loss,
        metrics=["accuracy"]
    )
    return model

# ===================================================
# 5. Espacio de hiperparámetros
# ===================================================
param_space = [
    {"conv_layers": 2, "filters": 32, "activation": "relu", "dense_layers": [64], "dropout": 0.2, "dropout_dense": 0.2, "optimizer": "adam"},
    {"conv_layers": 3, "filters": 16, "activation": "relu", "dense_layers": [128, 64], "dropout": 0.3, "dropout_dense": 0.3, "optimizer": "adam"},
    {"conv_layers": 2, "filters": 32, "activation": "tanh", "dense_layers": [64], "dropout": 0.2, "dropout_dense": 0.2, "optimizer": "rmsprop"}
]

# ===================================================
# 6. Entrenamiento y registro
# ===================================================
output_dim = len(np.unique(y))
epochs = 50
batch_size = 32

for params in param_space:
    run_name = f"CNN_{params['conv_layers']}conv_{params['activation']}"
    with mlflow.start_run(run_name=run_name):
        model = build_cnn(params, input_shape=X_train.shape[1:], output_dim=output_dim)
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
