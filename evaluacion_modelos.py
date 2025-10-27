import mlflow
import pandas as pd
from fontTools.ttLib.tables.S_V_G_ import doc_index_entry_format_0Size

from get_data import *
from Indices_fin import *
from Senales_cv import *
from backtesting import *
from standarizacion_datos import *
from backtesting import *
# 1. Cargar y preparar datos
df = get_data('ORCL')
df = calcular_indices(df, 30)
df_sen = generar_senales(df)
df = standarizacion_datos(df_sen)

# 2. Cargar modelo (usando el registry)

model_uri_cnn = 'file:///C:/Users/52331/PycharmProjects/PythonProject4/mlRUNS/663426234351669949/models/m-df8a6bc246dd412c96e393b2fe833e89/artifacts'
modelo_cnn = mlflow.pyfunc.load_model(model_uri_cnn)

model_uri_mlp = 'file:///C:/Users/52331/PycharmProjects/PythonProject4/mlRUNS/352987731800974246/models/m-444bec9eee394f94ace83b3318e13987/artifacts'
modelo_mlp = mlflow.pyfunc.load_model(model_uri_mlp)


# 4. Predicción
target_col = 'signal'


# División cronológica: 60% train, 20% test, 20% val
df_val = df.iloc[int(len(df) * 0.8):]

df_val_pred = df_val.drop(columns = 'signal')
X_val = df_val_pred.values
X_val_cnn = np.expand_dims(X_val, axis=-1)
p = modelo_mlp.predict(X_val)
df_sen = df_sen.iloc[int(len(df_sen) * 0.8):]
df_val_prd_mlp = df_sen.drop(columns = 'signal')


con = [
    p<0,
    p<0.6,
    p>0.61
]
ele = [
    -1,
    0,
    1
]
p_mlp = np.select(con, ele)
df_val_prd_mlp['signal'] = p_mlp
p = modelo_cnn.predict(X_val_cnn)
p_cnn = np.argmax(p, axis=1)
df_val_prd_cnn = df_sen.drop(columns = 'signal')
df_val_prd_cnn['signal'] = p_cnn

r_df_real, met_real, trd_real = backtest(df_sen,100000,0.125,n_shares =10,sl=0.05,tp=0.05,borrow_rate=0.0025,cash_threshold=0.2)
r_df_mlp, met_mlp, trd_mlp = backtest(df_val_prd_mlp,100000,0.125,n_shares =10,sl=0.05,tp=0.05,borrow_rate=0.0025,cash_threshold=0.2)
r_df_cnn, met_cnn, trd_cnn = backtest(df_val_prd_cnn,100000,0.125,n_shares =10,sl=0.05,tp=0.05,borrow_rate=0.0025,cash_threshold=0.2)
print(r_df_real)
print(r_df_mlp)
print(r_df_cnn)
print(met_real)
print(met_mlp)
print(met_cnn)
print(trd_real)
print(trd_mlp)
print(trd_cnn)
