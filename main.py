
from get_data import *
from Indices_fin import *
from Senales_cv import *

from standarizacion_datos import *
from modelo_cnn import *
from Modelos import *
from evaluacion_modelos import *
def main():
    print("Iniciando flujo de procesamiento...")

    # Parámetros generales del proceso
    w = 30
    ticker = 'ORCL'

    print("Obteniendo datos del activo:", ticker)
    df = get_data(ticker)

    print("Calculando índices financieros...")
    df = calcular_indices(df, w)

    print("Generando señales del trading system...")
    df_real = generar_senales(df)

    print("⚖️ Estandarizando datos...")
    df_estandar = standarizacion_datos(df_real)

    print(" Entrenando modelo MLP...")
    m_mlp = modelo_mlp(df_estandar, 'signal', 50, 32, 0.1, 0.6, 0.2,
                       'drift_report_mlp', w, 6, 3)

    print("Entrenando modelo CNN...")
    m_cnn = modelo_cnn(df_estandar, 'signal', 50, 32, 0.1, 0.6, 0.2,
                       'drift_report_cnn', w, 6, 3)

    print(" Cargando rutas a modelos (MLflow artifacts)...")
    ruta_mlp_ma = ('file:///C:/Users/52331/PycharmProjects/PythonProject4/'
                'Proyecto-3/mlruns/178340679539533950/models/'
                'm-0b8695dbd759460981fa0c73f11eeb48/artifacts')

    ruta_cnn_ma = ('file:///C:/Users/52331/PycharmProjects/PythonProject4/'
                'Proyecto-3/mlruns/560905997404354661/models/'
                'm-3183024883b64e599cdcdb94d5546fb7/artifacts')
    ruta_mlp_60 = 'file:///C:/Users/52331/PycharmProjects/PythonProject4/Proyecto-3/mlruns/178340679539533950/models/m-9586754481aa44b7be67983077e03c44/artifacts'
    ruta_cnn_60 ='file:///C:/Users/52331/PycharmProjects/PythonProject4/Proyecto-3/mlruns/560905997404354661/models/m-ecfa31c8ee4a423dae0ff9ff649cb89b/artifacts'

    print(" Evaluando modelo MLP...")
    r_df_real, met_real, r_df_mlp, met_mlp = eval_model(
        df_estandar, df_real, ruta_mlp_ma, 0.8, 'signal',
        100000, 0.0125, 100, 0.05, 0.05, 0.0025, 0.2, 1
    )
    print(" Evaluando modelo MLP con menor accuracy")
    _, _, r_df_mlp_60, met_mlp_60 = eval_model(
        df_estandar, df_real, ruta_mlp_60, 0.8, 'signal',
        100000, 0.0125, 100, 0.05, 0.05, 0.0025, 0.2, 1
    )

    print(" Evaluando modelo CNN...")
    _, _, r_df_cnn, met_cnn = eval_model(
        df_estandar, df_real, ruta_cnn_ma, 0.8, 'signal',
       100000, 0.0125, 100, 0.05, 0.05, 0.0025, 0.2, 2
   )
    print(" Evaluando modelo MLP con menor accuracy")
    _, _, r_df_cnn_60, met_cnn_60 = eval_model(
        df_estandar, df_real, ruta_cnn_60, 0.8, 'signal',
        100000, 0.0125, 100, 0.05, 0.05, 0.0025, 0.2, 2
    )

    print("Proceso completado con éxito.")
    print("Datos reales")
    print("   - datos reales:", met_real)
    print(" Resultados_mayor acurracy:")

    print("   - Métricas MLP:", met_mlp)
    print("   - Métricas CNN:", met_cnn)
    print("resultados con accuracy del 60%")
    print("   - Métricas MLP60:", met_mlp_60)
    print("   - Métricas CNN60:", met_cnn_60)

if __name__ == "__main__":
    main()